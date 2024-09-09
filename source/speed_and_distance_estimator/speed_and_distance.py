import cv2
from typing import List
import numpy as np
from utils import measure_distance, get_position_foot


class SpeedAndDistanceEstimator:
    def __init__(self):
        self.frame_window=5
        self.frame_rate=24

    def add_speed_and_distance_to_tracks(self, tracks):

        """
        Hàm này thêm thông tin về tốc độ và quãng đường đã di chuyển vào dữ liệu theo dõi.
        
        Args:
            tracks (Dict): Dữ liệu theo dõi của các đối tượng trên sân.
        Note:
            - Phương thức này bỏ qua các đối tượng "ball" và "referees".
            - Tốc độ được tính bằng km/h và khoảng cách bằng mét.
            - Thông tin tốc độ và khoảng cách được thêm vào mỗi khung hình trong tracks.
        
        """
        total_distance= {}

        for object, object_tracks in tracks.items():
            if object == "ball" or object == "referees":
                continue 
            number_of_frames = len(object_tracks)
            for frame_num in range(0,number_of_frames, self.frame_window):
                last_frame = min(frame_num+self.frame_window,number_of_frames-1 )

                for track_id,_ in object_tracks[frame_num].items():
                    if track_id not in object_tracks[last_frame]:
                        continue

                    start_position = object_tracks[frame_num][track_id]['position_transformed']
                    end_position = object_tracks[last_frame][track_id]['position_transformed']

                    if start_position is None or end_position is None:
                        continue
                    
                    distance_covered = measure_distance(start_position,end_position)
                    time_elapsed = (last_frame-frame_num)/self.frame_rate
                    speed_meteres_per_second = distance_covered/time_elapsed
                    speed_km_per_hour = speed_meteres_per_second*3.6

                    if object not in total_distance:
                        total_distance[object]= {}
                    
                    if track_id not in total_distance[object]:
                        total_distance[object][track_id] = 0
                    
                    total_distance[object][track_id] += distance_covered

                    for frame_num_batch in range(frame_num,last_frame):
                        if track_id not in tracks[object][frame_num_batch]:
                            continue
                        tracks[object][frame_num_batch][track_id]['speed'] = speed_km_per_hour
                        tracks[object][frame_num_batch][track_id]['distance'] = total_distance[object][track_id]

    def draw_speed_and_distance(self, frames: List[np.array], tracks) -> List[np.array]:
        """
        Vẽ thông tin về tốc độ và khoảng cách lên các khung hình.

        Args:
            frames (list): Danh sách các khung hình cần vẽ thông tin.
            tracks (dict): Từ điển chứa dữ liệu theo dõi của các đối tượng, bao gồm tốc độ và khoảng cách.

        Returns:
            list: Danh sách các khung hình đã được vẽ thông tin.

        Note:
            - Phương thức này bỏ qua các đối tượng "ball" và "referees".
            - Tốc độ và khoảng cách được vẽ gần chân của đối tượng.
        """
        output_frames = []
        for frame_num, frame in enumerate(frames):
            for object, object_tracks in tracks.items():
                if object == "ball" or object == "referees":
                    continue 
                for _, track_info in object_tracks[frame_num].items():
                    if "speed" in track_info:
                        speed = track_info.get('speed',None)
                        distance = track_info.get('distance',None)
                        if speed is None or distance is None:
                            continue
                       
                        bbox = track_info['bbox']
                        position = get_position_foot(bbox)
                        position = list(position)
                        position[1]+=40

                        position = tuple(map(int,position))


                        cv2.putText(
                            img=frame,
                            text=f"{speed:.2f} km/h",
                            org=position,
                            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                            fontScale=0.5,
                            color=(0,0,0),
                            thickness=2
                        )

                        cv2.putText(
                            img=frame,
                            text=f"{distance:.2f} m",
                            org=(position[0], position[1] + 20),
                            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                            fontScale=0.5,
                            color=(0,0,0),
                            thickness=2
                        )
                        # print("Done")
            output_frames.append(frame)
        return output_frames


