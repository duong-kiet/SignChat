import sys
import math
import numpy as np
import cv2
import os
import pickle
import gzip
import subprocess
import torch

from dtw import dtw
from constants import PAD_TOKEN

# Định nghĩa cấu trúc landmarks
FACE_LANDMARKS_GROUPS = {
    'silhouette': [10, 332, 389, 323, 397, 378, 152, 149, 172, 93, 162, 103],
    'leftEyebrowUpper': [300, 293, 334, 296, 336, 285],
    'leftEyebrowLower': [353, 276, 283, 282, 295],
    'rightEyebrowUpper': [70, 63, 105, 66, 107, 55],
    'rightEyebrowLower': [124, 46, 53, 52, 65],
    'leftEyeUpper0': [466, 388, 387, 386, 385, 384, 398],
    'leftEyeLower0': [263, 249, 390, 373, 374, 380, 381, 382, 362],
    'rightEyeUpper0': [246, 161, 160, 159, 158, 157, 173],
    'rightEyeLower0': [33, 7, 163, 144, 145, 153, 154, 155, 133],
    'lipsUpperOuter': [61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291],
    'lipsLowerOuter': [146, 91, 181, 84, 17, 314, 405, 321, 375],
    'lipsUpperInner': [78, 191, 80, 81, 82, 13, 312, 311, 310, 415, 308],
    'lipsLowerInner': [95, 88, 178, 87, 14, 317, 402, 318, 324],
}

# Chỉ số cho pose landmarks
# 11: vai trái, 12: vai phải, 13: khuỷu tay trái, 14: khuỷu tay phải, 
# 15: cổ tay trái, 16: cổ tay phải, 23: hông trái, 24: hông phải
POSE_LANDMARKS_INDICES = [15, 13, 11, 23, 24, 12, 14, 16]

# Chỉ số cho hand landmarks (1-20)
HAND_LANDMARKS_INDICES = list(range(1, 21))

# Offset để chỉ định index trong cấu trúc dữ liệu
POSE_OFFSET = 0  # Pose landmarks bắt đầu từ index 0
LEFT_HAND_OFFSET = len(POSE_LANDMARKS_INDICES)  # Left hand sau pose
RIGHT_HAND_OFFSET = LEFT_HAND_OFFSET + len(HAND_LANDMARKS_INDICES)  # Right hand sau left hand

# Plot a video given a tensor of joints, a file path, video name and references/sequence ID
def plot_video(joints,
               file_path,
               video_name,
               references=None,
               skip_frames=1,
               sequence_ID=None):
    # Create video template
    FPS = (25 // skip_frames)
    video_file = file_path + "/{}.mp4".format(video_name.split(".")[0])
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')

    if references is None:
        video = cv2.VideoWriter(video_file, fourcc, float(FPS), (650, 650), True)
    elif references is not None:
        video = cv2.VideoWriter(video_file, fourcc, float(FPS), (1300, 650), True)  # Long

    num_frames = 0

    for (j, frame_joints) in enumerate(joints):

        # Reached padding
        if PAD_TOKEN in frame_joints:
            continue

        # Initialise frame of white
        frame = np.ones((650, 650, 3), np.uint8) * 255

        # Bỏ counter ở cuối và điều chỉnh kích thước joints
        frame_joints = frame_joints[:-1] * 3

        # Chuyển đổi frame_joints thành dạng 2D để vẽ
        # Giả sử frame_joints có shape là (N*3,) với N là số lượng joints
        # và mỗi joints có 3 giá trị (x, y, z) hoặc (x, y, confidence)
        n_joints = len(frame_joints) // 3
        frame_joints_2d = np.reshape(frame_joints, (n_joints, 3))[:, :2]
        
        # Vẽ frame từ joints 2D
        draw_frame_2D(frame, frame_joints_2d)

        cv2.putText(frame, "Predicted Sign Pose", (180, 600), cv2.FONT_HERSHEY_SIMPLEX, 1,
                    (0, 0, 255), 2)

        # If reference is provided, create and concatenate on the end
        if references is not None:
            # Extract the reference joints
            ref_joints = references[j]
            # Initialise frame of white
            ref_frame = np.ones((650, 650, 3), np.uint8) * 255

            # Bỏ counter ở cuối và điều chỉnh kích thước
            ref_joints = ref_joints[:-1] * 3

            # Chuyển đổi thành dạng 2D
            ref_joints_2d = np.reshape(ref_joints, (n_joints, 3))[:, :2]

            # Vẽ joints lên frame
            draw_frame_2D(ref_frame, ref_joints_2d)

            cv2.putText(ref_frame, "Ground Truth Pose", (190, 600), cv2.FONT_HERSHEY_SIMPLEX, 1,
                        (0, 0, 0), 2)

            frame = np.concatenate((frame, ref_frame), axis=1)

            if sequence_ID:
                sequence_ID_write = "Sequence ID: " + sequence_ID.split("/")[-1]
                cv2.putText(frame, sequence_ID_write, (700, 635), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                            (0, 0, 0), 2)
        # Write the video frame
        video.write(frame)
        num_frames += 1
    # Release the video
    video.release()

# This is the format of the 3D data, outputted from the Inverse Kinematics model
def getSkeletalModelStructure():
    """
    Định nghĩa cấu trúc khung xương dạng tuple (điểm bắt đầu, điểm kết thúc, loại xương)
    """
    skeleton = []
    
    # Cấu trúc POSE (thân người và tay)
    # Vai trái (11) đến khuỷu tay trái (13)
    skeleton.append((POSE_OFFSET + 2, POSE_OFFSET + 1, 1))  # vai trái -> khuỷu tay trái
    # Khuỷu tay trái (13) đến cổ tay trái (15) 
    skeleton.append((POSE_OFFSET + 1, POSE_OFFSET + 0, 2))  # khuỷu tay trái -> cổ tay trái
    
    # Vai phải (12) đến khuỷu tay phải (14)
    skeleton.append((POSE_OFFSET + 5, POSE_OFFSET + 6, 1))  # vai phải -> khuỷu tay phải
    # Khuỷu tay phải (14) đến cổ tay phải (16)
    skeleton.append((POSE_OFFSET + 6, POSE_OFFSET + 7, 2))  # khuỷu tay phải -> cổ tay phải
    
    # Vai trái (11) đến hông trái (23)
    skeleton.append((POSE_OFFSET + 2, POSE_OFFSET + 3, 3))  # vai trái -> hông trái
    # Vai phải (12) đến hông phải (24)
    skeleton.append((POSE_OFFSET + 5, POSE_OFFSET + 4, 3))  # vai phải -> hông phải
    
    # Vai trái (11) đến vai phải (12)
    skeleton.append((POSE_OFFSET + 2, POSE_OFFSET + 5, 4))  # vai trái -> vai phải
    # Hông trái (23) đến hông phải (24)
    skeleton.append((POSE_OFFSET + 3, POSE_OFFSET + 4, 4))  # hông trái -> hông phải
    
    # Kết nối cổ tay trái (15) với bàn tay trái (điểm 0 của tay)
    skeleton.append((POSE_OFFSET + 0, LEFT_HAND_OFFSET + 0, 5))
    
    # Kết nối cổ tay phải (16) với bàn tay phải (điểm 0 của tay)
    skeleton.append((POSE_OFFSET + 7, RIGHT_HAND_OFFSET + 0, 5))
    
    # Cấu trúc BÀN TAY TRÁI
    # Xương ngón cái
    skeleton.append((LEFT_HAND_OFFSET + 0, LEFT_HAND_OFFSET + 1, 10))  # CMC -> MCP
    skeleton.append((LEFT_HAND_OFFSET + 1, LEFT_HAND_OFFSET + 2, 10))  # MCP -> IP
    skeleton.append((LEFT_HAND_OFFSET + 2, LEFT_HAND_OFFSET + 3, 10))  # IP -> TIP
    
    # Xương ngón trỏ
    skeleton.append((LEFT_HAND_OFFSET + 0, LEFT_HAND_OFFSET + 4, 11))  # Wrist -> MCP
    skeleton.append((LEFT_HAND_OFFSET + 4, LEFT_HAND_OFFSET + 5, 11))  # MCP -> PIP
    skeleton.append((LEFT_HAND_OFFSET + 5, LEFT_HAND_OFFSET + 6, 11))  # PIP -> DIP
    skeleton.append((LEFT_HAND_OFFSET + 6, LEFT_HAND_OFFSET + 7, 11))  # DIP -> TIP
    
    # Xương ngón giữa
    skeleton.append((LEFT_HAND_OFFSET + 0, LEFT_HAND_OFFSET + 8, 12))  # Wrist -> MCP
    skeleton.append((LEFT_HAND_OFFSET + 8, LEFT_HAND_OFFSET + 9, 12))  # MCP -> PIP
    skeleton.append((LEFT_HAND_OFFSET + 9, LEFT_HAND_OFFSET + 10, 12))  # PIP -> DIP
    skeleton.append((LEFT_HAND_OFFSET + 10, LEFT_HAND_OFFSET + 11, 12))  # DIP -> TIP
    
    # Xương ngón áp út
    skeleton.append((LEFT_HAND_OFFSET + 0, LEFT_HAND_OFFSET + 12, 13))  # Wrist -> MCP
    skeleton.append((LEFT_HAND_OFFSET + 12, LEFT_HAND_OFFSET + 13, 13))  # MCP -> PIP
    skeleton.append((LEFT_HAND_OFFSET + 13, LEFT_HAND_OFFSET + 14, 13))  # PIP -> DIP
    skeleton.append((LEFT_HAND_OFFSET + 14, LEFT_HAND_OFFSET + 15, 13))  # DIP -> TIP
    
    # Xương ngón út
    skeleton.append((LEFT_HAND_OFFSET + 0, LEFT_HAND_OFFSET + 16, 14))  # Wrist -> MCP
    skeleton.append((LEFT_HAND_OFFSET + 16, LEFT_HAND_OFFSET + 17, 14))  # MCP -> PIP
    skeleton.append((LEFT_HAND_OFFSET + 17, LEFT_HAND_OFFSET + 18, 14))  # PIP -> DIP
    skeleton.append((LEFT_HAND_OFFSET + 18, LEFT_HAND_OFFSET + 19, 14))  # DIP -> TIP
    
    # Cấu trúc BÀN TAY PHẢI - tương tự với bàn tay trái nhưng với offset khác
    # Xương ngón cái
    skeleton.append((RIGHT_HAND_OFFSET + 0, RIGHT_HAND_OFFSET + 1, 20))  # CMC -> MCP
    skeleton.append((RIGHT_HAND_OFFSET + 1, RIGHT_HAND_OFFSET + 2, 20))  # MCP -> IP
    skeleton.append((RIGHT_HAND_OFFSET + 2, RIGHT_HAND_OFFSET + 3, 20))  # IP -> TIP
    
    # Xương ngón trỏ
    skeleton.append((RIGHT_HAND_OFFSET + 0, RIGHT_HAND_OFFSET + 4, 21))  # Wrist -> MCP
    skeleton.append((RIGHT_HAND_OFFSET + 4, RIGHT_HAND_OFFSET + 5, 21))  # MCP -> PIP
    skeleton.append((RIGHT_HAND_OFFSET + 5, RIGHT_HAND_OFFSET + 6, 21))  # PIP -> DIP
    skeleton.append((RIGHT_HAND_OFFSET + 6, RIGHT_HAND_OFFSET + 7, 21))  # DIP -> TIP
    
    # Xương ngón giữa
    skeleton.append((RIGHT_HAND_OFFSET + 0, RIGHT_HAND_OFFSET + 8, 22))  # Wrist -> MCP
    skeleton.append((RIGHT_HAND_OFFSET + 8, RIGHT_HAND_OFFSET + 9, 22))  # MCP -> PIP
    skeleton.append((RIGHT_HAND_OFFSET + 9, RIGHT_HAND_OFFSET + 10, 22))  # PIP -> DIP
    skeleton.append((RIGHT_HAND_OFFSET + 10, RIGHT_HAND_OFFSET + 11, 22))  # DIP -> TIP
    
    # Xương ngón áp út
    skeleton.append((RIGHT_HAND_OFFSET + 0, RIGHT_HAND_OFFSET + 12, 23))  # Wrist -> MCP
    skeleton.append((RIGHT_HAND_OFFSET + 12, RIGHT_HAND_OFFSET + 13, 23))  # MCP -> PIP
    skeleton.append((RIGHT_HAND_OFFSET + 13, RIGHT_HAND_OFFSET + 14, 23))  # PIP -> DIP
    skeleton.append((RIGHT_HAND_OFFSET + 14, RIGHT_HAND_OFFSET + 15, 23))  # DIP -> TIP
    
    # Xương ngón út
    skeleton.append((RIGHT_HAND_OFFSET + 0, RIGHT_HAND_OFFSET + 16, 24))  # Wrist -> MCP
    skeleton.append((RIGHT_HAND_OFFSET + 16, RIGHT_HAND_OFFSET + 17, 24))  # MCP -> PIP
    skeleton.append((RIGHT_HAND_OFFSET + 17, RIGHT_HAND_OFFSET + 18, 24))  # PIP -> DIP
    skeleton.append((RIGHT_HAND_OFFSET + 18, RIGHT_HAND_OFFSET + 19, 24))  # DIP -> TIP
    
    return tuple(skeleton)

# Draw a line between two points, if they are positive points
def draw_line(im, joint1, joint2, c=(0, 0, 255), t=1, width=3):
    """
    Vẽ đường thẳng kết nối hai điểm khớp
    """
    thresh = -100
    if joint1[0] > thresh and joint1[1] > thresh and joint2[0] > thresh and joint2[1] > thresh:
        center = (int((joint1[0] + joint2[0]) / 2), int((joint1[1] + joint2[1]) / 2))
        length = int(math.sqrt(((joint1[0] - joint2[0]) ** 2) + ((joint1[1] - joint2[1]) ** 2))/2)
        angle = math.degrees(math.atan2((joint1[0] - joint2[0]),(joint1[1] - joint2[1])))
        cv2.ellipse(im, center, (width, length), -angle, 0.0, 360.0, c, -1)

# Draw the frame given 2D joints that are in the Inverse Kinematics format
def draw_frame_2D(frame, joints):
    """
    Vẽ khung xương 2D lên frame
    """
    # Đường kẻ giữa
    draw_line(frame, [1, 650], [1, 1], c=(0, 0, 0), t=1, width=1)
    
    # Offset để căn giữa khung xương
    offset = [350, 250]
    
    # Lấy cấu trúc khung xương
    skeleton = getSkeletalModelStructure()
    skeleton = np.array(skeleton)
    
    number = skeleton.shape[0]
    
    # Tăng kích thước và vị trí của các điểm khớp
    # Giả sử joints là mảng 2D với kích thước (num_joints, 2)
    total_joints = len(POSE_LANDMARKS_INDICES) + 2 * len(HAND_LANDMARKS_INDICES)
    
    # Đảm bảo rằng số lượng điểm khớp phù hợp
    assert joints.shape[0] >= total_joints, f"Expected at least {total_joints} joints, got {joints.shape[0]}"
    
    # Chỉnh kích thước joints
    joints = joints * 10 * 5  # Điều chỉnh hệ số phóng to
    joints = joints + np.ones((joints.shape[0], 2)) * offset
    
    # Vẽ các xương dựa trên cấu trúc khung xương
    for j in range(number):
        c = get_bone_colour(skeleton, j)
        
        start_idx = skeleton[j, 0]
        end_idx = skeleton[j, 1]
        
        # Kiểm tra xem các chỉ số có hợp lệ không
        if start_idx < joints.shape[0] and end_idx < joints.shape[0]:
            draw_line(frame, 
                     [joints[start_idx][0], joints[start_idx][1]],
                     [joints[end_idx][0], joints[end_idx][1]], 
                     c=c, t=1, width=2)

# get bone colour given index
def get_bone_colour(skeleton, j):
    """
    Xác định màu sắc cho từng loại xương
    """
    bone = skeleton[j, 2]
    
    # Màu cho POSE (thân người)
    if bone == 1:  # Vai đến khuỷu tay
        c = (0, 0, 255)  # Đỏ
    elif bone == 2:  # Khuỷu tay đến cổ tay
        c = (0, 128, 255)  # Cam
    elif bone == 3:  # Vai đến hông
        c = (0, 255, 0)  # Xanh lá
    elif bone == 4:  # Vai trái - vai phải, hông trái - hông phải
        c = (255, 0, 0)  # Xanh dương
    elif bone == 5:  # Cổ tay đến bàn tay
        c = (255, 0, 255)  # Tím
        
    # Màu cho BÀN TAY TRÁI
    elif bone == 10:  # Ngón cái trái
        c = (0, 204, 204)  # Vàng nhạt
    elif bone == 11:  # Ngón trỏ trái
        c = (51, 255, 51)  # Xanh lá nhạt
    elif bone == 12:  # Ngón giữa trái
        c = (255, 0, 0)  # Xanh dương
    elif bone == 13:  # Ngón áp út trái
        c = (204, 153, 255)  # Tím nhạt
    elif bone == 14:  # Ngón út trái
        c = (51, 255, 255)  # Vàng
        
    # Màu cho BÀN TAY PHẢI
    elif bone == 20:  # Ngón cái phải
        c = (0, 204, 204)  # Vàng nhạt
    elif bone == 21:  # Ngón trỏ phải
        c = (51, 255, 51)  # Xanh lá nhạt
    elif bone == 22:  # Ngón giữa phải
        c = (255, 0, 0)  # Xanh dương
    elif bone == 23:  # Ngón áp út phải
        c = (204, 153, 255)  # Tím nhạt
    elif bone == 24:  # Ngón út phải
        c = (51, 255, 255)  # Vàng
    else:
        c = (127, 127, 127)  # Xám cho các xương khác
        
    return c

# Apply DTW to the produced sequence, so it can be visually compared to the reference sequence
def alter_DTW_timing(pred_seq, ref_seq):
    """
    Áp dụng DTW để điều chỉnh timing của chuỗi dự đoán
    
    :param pred_seq: Chuỗi dự đoán
    :param ref_seq: Chuỗi tham chiếu
    :return: Chuỗi dự đoán đã điều chỉnh, chuỗi tham chiếu, điểm DTW
    """
    print("pred_seq:", pred_seq)

    # Định nghĩa hàm chi phí
    euclidean_norm = lambda x, y: np.sum(np.abs(x - y))

    # Cắt tham chiếu xuống giá trị đếm tối đa
    _, ref_max_idx = torch.max(ref_seq[:, -1], 0)
    if ref_max_idx == 0: ref_max_idx += 1
    # Cắt frames theo counter
    ref_seq = ref_seq[:ref_max_idx,:].cpu().numpy()

    # Cắt dự đoán xuống giá trị đếm tối đa
    _, hyp_max_idx = torch.max(pred_seq[:, -1], 0)
    if hyp_max_idx == 0: hyp_max_idx += 1
    # Cắt frames theo counter
    pred_seq = pred_seq[:hyp_max_idx,:].cpu().numpy()

    # Chạy DTW trên chuỗi tham chiếu và dự đoán
    d, cost_matrix, acc_cost_matrix, path = dtw(ref_seq[:,:-1], pred_seq[:,:-1], dist=euclidean_norm)

    # Chuẩn hóa chi phí DTW theo độ dài chuỗi
    d = d / acc_cost_matrix.shape[0]

    # Khởi tạo chuỗi mới
    new_pred_seq = np.zeros_like(ref_seq)
    # j theo dõi vị trí trong chuỗi tham chiếu
    j = 0
    skips = 0
    squeeze_frames = []
    for (i, pred_num) in enumerate(path[0]):
        if i == len(path[0]) - 1:
            break
            
        if path[1][i] == path[1][i + 1]:
            skips += 1
            
        # Nếu sắp có double
        if path[0][i] == path[0][i + 1]:
            squeeze_frames.append(pred_seq[i - skips])
            j += 1
        # Vừa kết thúc double
        elif path[0][i] == path[0][i - 1]:
            new_pred_seq[pred_num] = avg_frames(squeeze_frames)
            squeeze_frames = []
        else:
            new_pred_seq[pred_num] = pred_seq[i - skips]

    return new_pred_seq, ref_seq, d

# Find the average of the given frames
def avg_frames(frames):
    """
    Tính trung bình của các frames
    
    :param frames: Danh sách các frame cần tính trung bình
    :return: Frame trung bình
    """
    frames_sum = np.zeros_like(frames[0])
    for frame in frames:
        frames_sum += frame

    avg_frame = frames_sum / len(frames)
    return avg_frame