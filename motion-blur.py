import cv2
import numpy as np

def motion_blur(image, angle, strength):
    # 计算模糊核的大小
    kernel_size = int(strength * 2)
    
    # 创建一个水平方向的模糊核
    kernel = np.zeros((kernel_size, kernel_size))
    center = int(kernel_size / 2)
    slope_tan = np.tan(angle * np.pi / 180)
    slope_cot = 1 / slope_tan
    for i in range(kernel_size):
        offset = i - center
        if abs(offset * slope_tan) < kernel_size / 2:
            kernel[int(center - offset * slope_cot), i] = 1

    # 对图像进行滤波
    blurred_image = cv2.filter2D(image, -1, kernel)
    
    return blurred_image


def motion_blur2(image, angle, strength):
    # Convert the image to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    
    # Normalize the angle value
    angle = np.deg2rad(angle)
    
    # Compute kernel size based on strength
    kernel_size = int(strength * 2) + 1
    
    # Create an empty kernel
    kernel = np.zeros((kernel_size, kernel_size))
    
    # Compute the center of the kernel
    center = (kernel_size - 1) / 2
    
    # Compute the slope cotangent value
    slope_cot = np.tan(angle) / kernel_size
    
    # Generate the motion blur kernel
    for i in range(kernel_size):
        offset = abs(center - i)
        #kernel[int(center - offset * slope_cot), i] = 1
        idx = np.clip(int(center - offset * slope_cot), 0, kernel.shape[0] - 1)
        kernel[idx, i] = 1
    # Normalize the kernel
    kernel = kernel / np.sum(kernel)
    
    # Apply the motion blur kernel to the image
    motion_blurred_image = cv2.filter2D(gray_image, -1, kernel)
    
    rgb_frame = cv2.cvtColor(motion_blurred_image, cv2.COLOR_GRAY2RGB)
    
    return rgb_frame
    
    
    
    
# 读取连续XRGB帧数据
video_file = 'TQdst_DX11_176x144_BGRX8888_xvp2.rgb'
frame_width = 176
frame_height = 144
frame_count = 206

video_data = np.fromfile(video_file, dtype=np.uint8)
video_data = video_data.reshape(frame_count, frame_height, frame_width, 4)

# 进行运动模糊滤波
blurred_video = []
for frame in video_data:
    # 将XRGB转换为BGR格式
    bgr_frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2RGB)
    
    # 进行运动模糊滤波
    #blurred_frame = motion_blur(bgr_frame, 90, 5)
    blurred_frame = motion_blur2(bgr_frame, 90, 5)
    # 将BGR转换回XRGB格式
    xrgb_frame = cv2.cvtColor(blurred_frame, cv2.COLOR_RGB2BGRA)
    
    # 添加到滤波后的视频列表中
    blurred_video.append(xrgb_frame)

# 保存滤波后的连续XRGB帧数据
output_file = 'output.raw'
np.array(blurred_video).astype(np.uint8).tofile(output_file)