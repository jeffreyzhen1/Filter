import cv2
import numpy as np

# 定义每帧的大小和帧之间的间隔（假设每帧大小为frame_size字节，帧之间间隔为frame_interval字节）
width = 176
height = 144
frame_size = width*height*4
frame_interval = 0



def denoise1(frame):
    # 使用均值滤波对帧进行降噪处理
    denoised_frame = cv2.blur(frame, (3, 3))#3，3-》5，5
    return denoised_frame
    
def denoise2(frame):
    # 将裸数据图像帧转换为RGB格式
    frame_rgb = cv2.cvtColor(frame.reshape(height, width, 4), cv2.COLOR_BGRA2RGB)

    # 使用高斯模糊进行降噪处理
    denoised_frame = cv2.GaussianBlur(frame_rgb, (3, 3), 0)

    # 将RGB格式的图像帧转换回XRGB格式
    denoised_frame_xrgb = cv2.cvtColor(denoised_frame, cv2.COLOR_RGB2BGRA)
    return denoised_frame_xrgb


def denoise3(frame):
    # 将帧图像数据解析为XRGB图像格式
    frame_rgb = cv2.cvtColor(frame.reshape(height, width, 4), cv2.COLOR_BGRA2RGB)

    # 对图像应用去块滤波器
    deblocked_frame = cv2.fastNlMeansDenoisingColored(frame_rgb, None, 8, 8, 5, 15)#10, 10, 7, 21

    # 对滤波后的图像数据进行边缘增强
    enhanced_frame = cv2.detailEnhance(deblocked_frame, sigma_s=0.5, sigma_r=0.15)
    enhanced_frame_xrgb = cv2.cvtColor(enhanced_frame, cv2.COLOR_RGB2BGRA)
    return enhanced_frame_xrgb
        
def denoise4(frame):
    # 将裸数据图像帧转换为RGB格式
    frame_rgb = cv2.cvtColor(frame.reshape(height, width, 4), cv2.COLOR_BGRA2RGB)

    # 使用双边滤波
    denoised_frame = cv2.bilateralFilter(frame_rgb, 5, 180, 180)#xvp2 114

    # 将RGB格式的图像帧转换回XRGB格式
    denoised_frame_xrgb = cv2.cvtColor(denoised_frame, cv2.COLOR_RGB2BGRA)
    return denoised_frame_xrgb

    
# 从文件中读取连续的XRGB裸数据帧
with open('TQdst_DX11_176x144_BGRX8888_xvp2.rgb', 'rb') as file:
    frames_data = file.read()


# 将连续的XRGB裸数据帧分割为单独的帧
frames = []
num_frames = len(frames_data) // (frame_size + frame_interval)
for i in range(num_frames):
    start = i * (frame_size + frame_interval)
    end = start + frame_size
    frame_data = frames_data[start:end]
    frame = np.frombuffer(frame_data, dtype=np.uint8).reshape((height, width, 4))
    frames.append(frame)

# 进行降噪处理
filtered_frames = []
for frame in frames:
    filtered_frame = denoise4(frame)
    filtered_frames.append(filtered_frame)

# 将降噪处理后的帧保存为连续的XRGB裸数据
filtered_frames_data = b''.join([frame.tobytes() for frame in filtered_frames])
with open('filtered_frames.raw', 'wb') as file:
    file.write(filtered_frames_data)