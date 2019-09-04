
import torch
from torch.autograd import Variable
import argparse
import numpy as np
from os.path import *

import cv2

from flownet2 import models, losses, datasets
from flownet2.utils import tools

parser = argparse.ArgumentParser()

parser.add_argument('--start_epoch', type=int, default=1)
parser.add_argument('--total_epochs', type=int, default=10000)
parser.add_argument('--batch_size', '-b', type=int, default=8, help="Batch size")
parser.add_argument('--train_n_batches', type=int, default=-1,
                    help='Number of min-batches per epoch. If < 0, it will be determined by training_dataloader')
parser.add_argument('--crop_size', type=int, nargs='+', default=[256, 256],
                    help="Spatial dimension to crop training samples for training")
parser.add_argument('--gradient_clip', type=float, default=None)
parser.add_argument('--schedule_lr_frequency', type=int, default=0, help='in number of iterations (0 for no schedule)')
parser.add_argument('--schedule_lr_fraction', type=float, default=10)
parser.add_argument("--rgb_max", type=float, default=255.)

parser.add_argument('--number_workers', '-nw', '--num_workers', type=int, default=8)
parser.add_argument('--number_gpus', '-ng', type=int, default=-1, help='number of GPUs to use')
parser.add_argument('--no_cuda', action='store_true')

parser.add_argument('--seed', type=int, default=1)
parser.add_argument('--name', default='run', type=str, help='a name to append to the save directory')
parser.add_argument('--save', '-s', default='./work', type=str, help='directory for saving')

parser.add_argument('--validation_frequency', type=int, default=5, help='validate every n epochs')
parser.add_argument('--validation_n_batches', type=int, default=-1)
parser.add_argument('--render_validation', action='store_true',
                    help='run inference (save flows to file) and every validation_frequency epoch')

parser.add_argument('--inference', action='store_true')
parser.add_argument('--inference_size', type=int, nargs='+', default=[-1, -1],
                    help='spatial size divisible by 64. default (-1,-1) - largest possible valid size would be used')
parser.add_argument('--inference_batch_size', type=int, default=1)
parser.add_argument('--inference_n_batches', type=int, default=-1)
parser.add_argument('--save_flow', action='store_true', help='save predicted flows to file')

parser.add_argument('--resume', default='', type=str, metavar='PATH', help='path to latest checkpoint (default: none)')
parser.add_argument('--log_frequency', '--summ_iter', type=int, default=1, help="Log every n batches")

parser.add_argument('--skip_training', action='store_true')
parser.add_argument('--skip_validation', action='store_true')

parser.add_argument('--fp16', action='store_true', help='Run model in pseudo-fp16 mode (fp16 storage fp32 math).')
parser.add_argument('--fp16_scale', type=float, default=1024.,
                    help='Loss scaling, positive power of 2 values can improve fp16 convergence.')

tools.add_arguments_for_module(parser, models, argument_for_class='model', default='FlowNet2')

tools.add_arguments_for_module(parser, losses, argument_for_class='loss', default='L1Loss')

tools.add_arguments_for_module(parser, torch.optim, argument_for_class='optimizer', default='Adam',
                               skip_params=['params'])

tools.add_arguments_for_module(parser, datasets, argument_for_class='training_dataset', default='MpiSintelFinal',
                               skip_params=['is_cropped'],
                               parameter_defaults={'root': './MPI-Sintel/flow/training'})

tools.add_arguments_for_module(parser, datasets, argument_for_class='validation_dataset', default='MpiSintelClean',
                               skip_params=['is_cropped'],
                               parameter_defaults={'root': './MPI-Sintel/flow/training',
                                                   'replicates': 1})

tools.add_arguments_for_module(parser, datasets, argument_for_class='inference_dataset', default='MpiSintelClean',
                               skip_params=['is_cropped'],
                               parameter_defaults={'root': './MPI-Sintel/flow/training',
                                                   'replicates': 1})

args = parser.parse_args()
if args.number_gpus < 0 : args.number_gpus = torch.cuda.device_count()
parser.add_argument('--IGNORE',  action='store_true')
defaults = vars(parser.parse_args(['--IGNORE']))

args.model_class = tools.module_to_dict(models)[args.model]
args.optimizer_class = tools.module_to_dict(torch.optim)[args.optimizer]
args.loss_class = tools.module_to_dict(losses)[args.loss]

args.training_dataset_class = tools.module_to_dict(datasets)[args.training_dataset]
args.validation_dataset_class = tools.module_to_dict(datasets)[args.validation_dataset]
args.inference_dataset_class = tools.module_to_dict(datasets)[args.inference_dataset]

args.cuda = not args.no_cuda and torch.cuda.is_available()
args.log_file = join(args.save, 'args.txt')

args.grads = {}

if args.inference:
    args.skip_validation = True
    args.skip_training = True
    args.total_epochs = 1
    args.inference_dir = "{}/inference".format(args.save)


args.effective_batch_size = args.batch_size * args.number_gpus
args.effective_inference_batch_size = args.inference_batch_size * args.number_gpus
args.effective_number_workers = args.number_workers * args.number_gpus
gpuargs = {'num_workers': args.effective_number_workers,
           'pin_memory': True,
           'drop_last' : True} if args.cuda else {}
inf_gpuargs = gpuargs.copy()
inf_gpuargs['num_workers'] = args.number_workers


class FlowController:
    def __init__(self, model_path="flownet2/FlowNet2_checkpoint.pth.tar"):
        self.model = models.FlowNet2(args)
        checkpoint = torch.load(model_path)
        self.model.load_state_dict(checkpoint['state_dict'])
        self.model.eval()

        if torch.cuda.is_available():
            self.model.cuda()

        self.is_cropped = False

    @staticmethod
    def convert_flow_to_image(flow):
        image_shape = flow.shape[0:2] + (3,)

        hsv = np.zeros(shape=image_shape, dtype=np.uint8)
        hsv[..., 1] = 255
        mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        hsv[..., 0] = ang * 180 / np.pi / 2
        normalized_mag = np.asarray(np.clip(mag*40, 0, 255), dtype=np.uint8)
        hsv[..., 2] = normalized_mag
        rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
        rgb = np.asarray(rgb, np.uint8)
        return rgb

    def convert_video_to_flow(self, video_path, output_path="out", downsample_res=None, raw_save=False):
        video = cv2.VideoCapture(video_path)
        ret, prev_frame = video.read()
        if downsample_res is not None:
            prev_frame = cv2.resize(prev_frame, downsample_res)

        view_shape = list(prev_frame.shape[0:2])
        if not raw_save:
            view_shape[0] *= 2

        out_video = cv2.VideoWriter(output_path+".avi", cv2.VideoWriter_fourcc('M','J','P','G'), 24, tuple(view_shape))

        while video.isOpened():
            ret, frame = video.read()
            if ret == True:
                if downsample_res is not None:
                    frame = cv2.resize(frame, downsample_res)
                opt_flow = self.predict(frame, prev_frame)
                opt_flow_image = self.convert_flow_to_image(opt_flow)
                prev_frame = frame

                joint_image = np.append(frame, opt_flow_image, axis=1)
                cv2.imshow("FlowNet2", joint_image)

                if raw_save:
                    out_video.write(opt_flow_image)
                else:
                    out_video.write(joint_image)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            else:
                break

        video.release()
        out_video.release()
        cv2.destroyAllWindows()

    @staticmethod
    def preprocess_frames(frame1, frame2):
        assert frame1.shape == frame2.shape, "Shapes of both frames must be same"

        # Downscale image resolution to closest factor for 64, if smaller than 64 than upscale to 64
        # This part basically calculates which resolution it should scale the image to
        process_resolution = tuple([max(64 * (frame1.shape[i] // 64), 64) for i in range(2)])
        images = [cv2.resize(frame1, process_resolution), cv2.resize(frame2, process_resolution)]
        images = np.expand_dims(np.array(images).transpose(3, 0, 1, 2), axis=0)
        images = torch.from_numpy(images.astype(np.float32))

        return [images], [torch.zeros(images.size()[0:1] + (2,) + images.size()[-2:])]

    def predict(self, image1, image2):
        (data, target) = self.preprocess_frames(image1, image2)
        if args.cuda:
            data, target = [d.cuda() for d in data], [t.cuda() for t in target]
        data, target = [Variable(d) for d in data], [Variable(t) for t in target]

        with torch.no_grad():
            output = self.model(data[0])

        flow = cv2.resize(output.data.cpu().numpy()[0].transpose(1, 2, 0), (image1.shape[1], image1.shape[0]))
        return flow
