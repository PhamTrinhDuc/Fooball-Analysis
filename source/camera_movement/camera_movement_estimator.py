import pickle
import cv2
import os
import numpy as np
from typing import List
from configs.config import LoadConfig
from utils.bbox_utils import measure_distance, measure_xy_distance
APP_CONFIG = LoadConfig() 

class CameraMovementEstimator:
    """
    ước tính chuyển động của camera bằng cách so sánh vị trí của các điểm đặc trưng giữa các khung hình liên tiếp. 
    Sử dụng optical flow để theo dõi các điểm này và xác định chuyển động lớn nhất như là chuyển động của camera. 
    Kết quả là một danh sách các vector chuyển động cho mỗi khung hình.
    """
    def __init__(self, frame: np.array):

        self.MINIMUM_THRESHOLD = 5

        self.lk_params = dict(
            winSize = (15,15),
            maxLevel = 2,
            criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,10,0.03)
        )

        first_frame_grayscale = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        mask_features = np.zeros_like(first_frame_grayscale)
        mask_features[:,0:20] = 1
        mask_features[:,900:1050] = 1

        self.features = dict(
            maxCorners = 100,
            qualityLevel = 0.3,
            minDistance =3,
            blockSize = 7,
            mask = mask_features
        )
    
    def get_camera_movement(self, frames: List[np.array]) -> List[List[int]] :
        if os.path.exists(APP_CONFIG.camera_movement_path):
            with open(APP_CONFIG.camera_movement_path, "rb") as file:
                camera_movement = pickle.load(file)
                return camera_movement

        camera_movement = [[0,0]]*len(frames)

        old_gray = cv2.cvtColor(frames[0], cv2.COLOR_BGR2GRAY)
        """ Được sử dụng để tìm các điểm đặc trưng (góc) trong ảsnh. 
            Trả về một mảng các điểm góc tìm được trong ảnh. -> np.array(N điểm, 1, 2)"""
        old_features = cv2.goodFeaturesToTrack(old_gray,**self.features)

        for frame_num in range(1,len(frames)):
            frame_gray = cv2.cvtColor(frames[frame_num],cv2.COLOR_BGR2GRAY)

            """ Hàm này tính toán optical flow cho một tập các điểm sử dụng phương pháp Lucas-Kanade theo kiểu pyramid.
                new_features: Vị trí mới của các điểm trong ảnh thứ hai
                status: Mảng chỉ ra điểm nào được tìm thấy trong ảnh mới
                err: Mảng lỗi cho mỗi điểm"""
            
            new_features, status, error = cv2.calcOpticalFlowPyrLK(
                old_gray, # Ảnh đầu tiên (ảnh xám)
                frame_gray, # Ảnh thứ hai (ảnh xám)
                old_features, # Các điểm cần theo dõi từ ảnh đầu tiên
                None, **self.lk_params)

            max_distance = 0
            camera_movement_x, camera_movement_y = 0,0

            for i, (new,old) in enumerate(zip(new_features,old_features)):
                new_features_point = new.ravel()
                old_features_point = old.ravel()

                distance = measure_distance(new_features_point, old_features_point)
                if distance > max_distance:
                    max_distance = distance
                    camera_movement_x,camera_movement_y = measure_xy_distance(old_features_point, new_features_point ) 
            
            if max_distance > self.MINIMUM_THRESHOLD:
                camera_movement[frame_num] = [camera_movement_x,camera_movement_y]
                old_features = cv2.goodFeaturesToTrack(frame_gray, **self.features)

            old_gray = frame_gray.copy()
        
        directory = os.path.dirname(APP_CONFIG.camera_movement_path)
        os.makedirs(directory, exist_ok=True)
        with open(APP_CONFIG.camera_movement_path, "wb") as f:
            pickle.dump(camera_movement, f)

        return camera_movement
    

    def draw_camera_movement(self, frames: List[np.array], camera_movement_per_frame: List[List[int]]) -> List[np.array]:
        output_frames = []

        for frame_number, frame in enumerate(frames):
            overlay = frame.copy()

            cv2.rectangle(img=overlay,
                          pt1=(0,0),
                          pt2=(500,100),
                          color=(255,255,255),
                          thickness=-1)
            ALPHA = 0.6
            cv2.addWeighted(src1=overlay, alpha=ALPHA, 
                            src2=frame, beta=1 - ALPHA, 
                            gamma=0, dst=frame)
            x_movement, y_movement = camera_movement_per_frame[frame_number]
            frame = cv2.putText(img=frame, text=f"Camera Movement X: {x_movement:.2f}",
                                org=(10,30), 
                                fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                                fontScale=1,
                                color=(0,0,0),
                                thickness=3)
            frame = cv2.putText(img=frame, text=f"Camera Movement Y: {y_movement:.2f}",
                                org=(10,60), 
                                fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                                fontScale=1,
                                color=(0,0,0),
                                thickness=3)

            output_frames.append(frame) 
        return output_frames
