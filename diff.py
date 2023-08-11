import numpy as np

def calculate_max_abs_diff(file1, file2, width, height):
    frame_size = width * height * 4
    max_diff_list = []
    max_diff_pos_list = []
    with open(file1, 'rb') as f1, open(file2, 'rb') as f2:
        while True:
            frame_data1 = f1.read(frame_size)
            frame_data2 = f2.read(frame_size)
            if not frame_data1 or not frame_data2:
                break
            frame1 = np.frombuffer(frame_data1, dtype=np.uint8)
            frame2 = np.frombuffer(frame_data2, dtype=np.uint8)
            frame1 = frame1.reshape((4, height, width))
            frame2 = frame2.reshape((4, height, width))
            #max_abs_diff = np.max(np.abs(frame1 - frame2))
            #max_abs_diff_list.append(max_abs_diff)
            # abs_diff = np.abs(frame1 - frame2)
            # if np.any(abs_diff != 0):
                # max_abs_diff = np.max(abs_diff)
                # max_abs_diff_list.append(max_abs_diff)
                # max_diff_pos_list.append(np.argmax(abs_diff))
            diff = np.maximum(frame1, frame2) - np.minimum(frame1, frame2)
            max_diff = np.max(diff)
            max_diff_list.append(max_diff)
            max_diff_pos_list.append(np.argmax(diff))
    return max_diff_list, max_diff_pos_list

# 用法示例
file1 = 'inter_xvp2.rgb'
#file2 = 'TQdst_DX11_176x144_BGRX8888_xvp2.rgb'
file2 = 'filtered_frames.raw'
width = 176
height = 144
count = 0
max_diff_list, max_diff_pos_list = calculate_max_abs_diff(file1, file2, width, height)
for i, max_diff in enumerate(max_diff_list):
    max_diff_position = max_diff_pos_list[i]
    if(max_diff < 90):
        count = count + 1
    print("Frame", i+1, "Max absolute difference:", max_diff)
    print("Max difference position:", max_diff_position)
print("count < 90 :", count)