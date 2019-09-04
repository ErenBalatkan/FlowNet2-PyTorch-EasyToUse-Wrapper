from Flownet2Controller import FlowController
import cv2
import numpy as np


def main():
    # This assumes FlowNet2_checkpoint.pth.tar is placed directly under flownet2 folder
    # You can pass in model_path to the constructor if it is located elsewhere
    flow_controller = FlowController()

    flow_controller = FlowController("./flownet2/FlowNet2_checkpoint.pth.tar")

    # Prediction given 2 images
    im1 = cv2.imread("im1.png")
    im2 = cv2.imread("im2.png")

    flow = flow_controller.predict(im1, im2)
    # Important note : All predictions are made at maximum viable resolution to ensure prediction quality is high,
    # but this comes at a massive hit to performance, if you want fast executions I suggest downsampling images first

    # Can convert flow to image using built in method
    flow_image = flow_controller.convert_flow_to_image(flow)

    cv2.imshow("Random flow image", flow_image)
    cv2.waitKey()
    cv2.destroyAllWindows()

    # Can also convert video's to their optical flow variants using following method
    # Use raw=true argument for only saving the optical flow video
    # Set downsample_res if you want to process video faster
    flow_controller.convert_video_to_flow("cp77cinematic.mp4", "output", downsample_res=(320, 320))


if __name__ == "__main__":
    main()
