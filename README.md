# FlowNet2-Pytorch-EasyToUse-Wrapper
This repository provides a wrapper with very easy usage for Nvidia's FlowNet2 Implementation

# Test on Cyberpunk 2077 Cinematic
[![IMAGE ALT TEXT](http://img.youtube.com/vi/6GXBBtCxihM/0.jpg)](https://www.youtube.com/watch?v=6GXBBtCxihM "Video Title")

# Installation
* Step 1 :  
Start with Nvidia's Installation guide located at https://github.com/NVIDIA/flownet2-pytorch

* Step 2 :  
Place all the files under flownet2-pytorch-master to flownet2 folder. (Or you can simply rename flownet2-pytorch-master directory to flownet2 and place it directly on root directory)

# Usage
Please refer to how_to_use.py for code examples

```python

 # For predicting flow given 2 images
flow_controller.predict(im1, im2)

# For converting flow matrix(output of predict method) into image 
flow_controller.convert_flow_to_image(flow) 

# For converting videos located on disk to optical flow videos
flow_controller.convert_video_to_flow("cp77cinematic.mp4", "output", downsample_res=(320, 320)) 
```
