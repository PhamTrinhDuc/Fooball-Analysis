import pickle
import cv2
import os
import numpy as np
import pandas as pd
from supervision import ByteTrack, Detections
from typing import List, Tuple, Dict
from utils import get_center, get_bbox_width, get_position_foot, save_data, load_data
from configs.config import APP_CFG


class Tracker:
    def __init__(self):
        self.model = APP_CFG.load_model()
        self.tracker = ByteTrack()
    
    def detect_frames(self, frames: list[np.array]) -> List[Detections]:
        """
        Args:
            frames: (List[np.array, shape=(H, W, 3)]) list of frames to detect objects
        Returns:
            detections: (List[Detections]) list of detected for each frame
        """
        BATCH_SIZE=20 
        detections = [] 
        for i in range(0,len(frames),BATCH_SIZE):
            detections_batch = self.model.predict(frames[i:i+BATCH_SIZE],conf=0.1)
            detections += detections_batch
        return detections
     

    def get_objects_track(self, frames: List[np.array]) -> Dict[str, List[Dict[int, Dict[str, List[float]]]]]:
        """
        Args:  
            frames: (List[Tuple]) list of frames to track objects
        Returns:
            tracks: (Dict[str, List[Dict[int, Dict[str, List[float]]]]) dictionary containing tracks of players, referees and balls
        """

        if os.path.exists(APP_CFG.frames_path):
            tracks = load_data(APP_CFG.frames_path)
            return tracks
        
        detections = self.detect_frames(frames)

        tracks = {
            "players": [],
            "referees": [],
            "ball": []
        }

        for frame_num, detection in enumerate(detections):
            cls_names = detection.names # {0: 'Goalkeeper', 1: 'Player' ...}
            cls_names_inv = {v:k for k,v in cls_names.items()} # {'Goalkeeper': 0, 'Player': 1 ...}

            # Covert to supervision Detection format
            detection_supervision = Detections.from_ultralytics(detection)

            # Convert GoalKeeper to player object
            # class_id=array([3, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 2, 2, 2, 2, 2, 2, 3, 1, 0, 2])
            for object_ind , class_id in enumerate(detection_supervision.class_id):
                if cls_names[class_id] == "goalkeeper":
                    detection_supervision.class_id[object_ind] = cls_names_inv["player"]

            # Track Objects
            detection_with_tracks = self.tracker.update_with_detections(detection_supervision)
            # print(detection_with_tracker[0]) Detections(xyxy=array([[1852.2, 808.25,1893.1,917.86]], dtype=float32), mask=None, confidence=array([0.93925], dtype=float32), class_id=array([3]), tracker_id=array([1]), data={'class_name': array(['referee'], dtype='<U10')})
            
            tracks["players"].append({})
            tracks["referees"].append({})
            tracks["ball"].append({})

            for frame_detection in detection_with_tracks:
                bbox = frame_detection[0].tolist()
                cls_id = frame_detection[3]
                track_id = frame_detection[4]

                if cls_id == cls_names_inv['player']:
                    tracks["players"][frame_num][track_id] = {"bbox":bbox}
                
                if cls_id == cls_names_inv['referee']:
                    tracks["referees"][frame_num][track_id] = {"bbox":bbox}
            
            for frame_detection in detection_supervision:
                bbox = frame_detection[0].tolist()
                cls_id = frame_detection[3]

                if cls_id == cls_names_inv['ball']:
                    tracks["ball"][frame_num][1] = {"bbox":bbox}

        
        directory = os.path.dirname(APP_CFG.frames_path)
        os.makedirs(directory, exist_ok=True)
        save_data(APP_CFG.frames_path, tracks)
        return tracks
    
    def draw_ellipse(self,frame: np.array, bbox: List[float], color: Tuple[int, int, int], track_id: int = None) -> np.array:
        """
        Args:
            frame: (np.array, shape=(H, W, 3)) frame to draw on 
            color: (Tuple[int, int, int]) color of the ellipse
            bbox: (List[float], len(bbox) == 4) bounding box of the player object
            track_id: (int) id of the object
        Returns:
            frame with the ellipse drawn on it
        """
        y2 = int(bbox[3])
        x_center, _ = get_center(bbox)
        width = get_bbox_width(bbox)

        cv2.ellipse(
            frame,
            center=(x_center,y2),
            axes=(int(width), int(0.35*width)),
            angle=0.0,
            startAngle=-45,
            endAngle=235,
            color = color,
            thickness=2,
            lineType=cv2.LINE_4
        )

        rectangle_width = 40
        rectangle_height=20
        x1_rect = x_center - rectangle_width//2
        x2_rect = x_center + rectangle_width//2
        y1_rect = (y2- rectangle_height//2) +15
        y2_rect = (y2+ rectangle_height//2) +15

        if track_id is not None:
            cv2.rectangle(img=frame,
                          pt1=(int(x1_rect),int(y1_rect) ),
                          pt2=(int(x2_rect),int(y2_rect)),
                          color=color,
                          thickness=cv2.FILLED)
            
            x1_text = x1_rect+12
            if track_id > 99:
                x1_text -=10
            
            cv2.putText(
                img=frame,
                text=f"{track_id}",
                org=(int(x1_text),int(y1_rect+15)),
                fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=0.6,
                color=(0,0,0),
                thickness=2
            )

        return frame
    
    def draw_triangle(self,frame: np.array, bbox: List[float], color: Tuple[int, int, int]) -> np.array:
        """
        Args:
            frame: (np.array, shape=(H, W, 3)) frame to draw on 
            color: (Tuple[int, int, int]) color of the triangle
            bbox: (List[float], len(bbox) == 4) bounding box of the ball object
        Returns:
            frame with the triangle drawn on it
        """
        y= int(bbox[1])
        x,_ = get_center(bbox)

        triangle_points = np.array([
            [x,y],
            [x-10,y-20],
            [x+10,y-20],
        ])
        cv2.drawContours(frame, [triangle_points],0,color, cv2.FILLED)
        cv2.drawContours(frame, [triangle_points],0,(0,0,0), 2)

        return frame
    
    def draw_annotations(self, video_frames: List[np.array], tracks, team_ball_control: List[int]) -> List[np.array]:
        """
        Args:
            tracks: (Dict[str, List[Dict[int, Dict[str, List[float]]]]) dictionary containing tracks of players, referees and balls
            frames: (List[np.array, shape=(H, W, 3)]) list of frames to draw on
        Returns:
            frames_output: (List[np.array, shape=(H, W, 3)]) list of frames with annotations drawn on them
        """
        output_video_frames= []
        for frame_num, frame in enumerate(video_frames):
            frame = frame.copy()

            player_dict = tracks["players"][frame_num]
            ball_dict = tracks["ball"][frame_num]
            referee_dict = tracks["referees"][frame_num]

            # Draw Players
            for track_id, player in player_dict.items():
                color = player.get("team_color",(0,0,255))
                frame = self.draw_ellipse(frame, player["bbox"],color, track_id)

                if player.get('has_ball',False):
                    frame = self.draw_triangle(frame, player["bbox"],(0,0,255))

            # Draw Referee
            for _, referee in referee_dict.items():
                frame = self.draw_ellipse(frame, referee["bbox"],(0,255,255))
            
            # Draw ball 
            for track_id, ball in ball_dict.items():
                frame = self.draw_triangle(frame, ball["bbox"],(0,255,0))

            # Draw team control
            frame = self.draw_team_ball_control(frame, frame_num, team_ball_control)
            

            output_video_frames.append(frame)

        return output_video_frames
    
    def interpolate_ball_positions(self, ball_positions: List[Dict[int, Dict[str, List[float]]]]) -> List[Dict[int, Dict[str, List[float]]]]:
        """
            Args:
                ball_positions: (List[Dict[int, Dict[str, List[float]]]]) list of ball positions
            Returns:
                ball_positions: (List[Dict[int, Dict[str, List[float]]]]) list of ball positions with missing values interpolated
        """
        
        ball_positions = [x.get(1,{}).get('bbox',[]) for x in ball_positions]
        df_ball_positions = pd.DataFrame(ball_positions,columns=['x1','y1','x2','y2'])

        # Interpolate missing values
        df_ball_positions = df_ball_positions.interpolate()
        df_ball_positions = df_ball_positions.bfill()

        ball_positions = [{1: {"bbox":x}} for x in df_ball_positions.to_numpy().tolist()]

        return ball_positions
    
    def draw_team_ball_control(self, frame: np.array, frame_num: int, team_ball_control: np.array) -> np.array :
        overlap = frame.copy()
        cv2.rectangle(
            img=overlap,
            pt1=(1350, 850),
            pt2=(1900, 970),
            color=(255, 255, 255),
            thickness=-1
        )

        ALPHA = 0.4

        cv2.addWeighted(
            src1=overlap,
            alpha=ALPHA,
            src2=frame,
            beta=1 - ALPHA,
            gamma=0,
            dst=frame
        )

        team_ball_control_till_frame = team_ball_control[:frame_num+1]
        team_control_1 = team_ball_control_till_frame[team_ball_control_till_frame==1].shape[0]
        team_control_2 = team_ball_control_till_frame[team_ball_control_till_frame==2].shape[0]
        team_1 = team_control_1 / (team_control_1 + team_control_2)
        team_2 = team_control_2 / (team_control_1 + team_control_2)

        cv2.putText( 
            img=frame,
            text = f"Team 1 Ball Control: {team_1 * 100 :.2f}%",
            org=(1400, 900),
            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=1,
            color=(0,0,0),
            thickness=3
        )

        cv2.putText(
            img=frame,
            text = f"Team 2 Ball Control: {team_2 * 100 :.2f}%",
            org=(1400, 950),
            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=1,
            color=(0,0,0),
            thickness=3
        )

        return frame
    
    def add_possition_to_tracks(self, tracks):
        for object, object_track in tracks.items():
            for frame_num, frame in enumerate(object_track):
                for track_id, track_info in frame.items():
                    bbox = track_info['bbox']
                    if object == 'ball':
                        possition = get_center(bbox)
                        tracks[object][frame_num][track_id]['possition'] = possition
                    else:
                        possition_foot = get_position_foot(bbox)
                        tracks[object][frame_num][track_id]['possition'] = possition_foot

    def adjust_position_to_tracks(self, tracks, camera_movement_per_frame: List[List[int]]):
        for object, object_track in tracks.items():
            for frame_num, frame in enumerate(object_track):
                for track_id, track_info in frame.items():
                    possition = track_info['possition']
                    camera_movement = camera_movement_per_frame[frame_num]
                    possition_adjusted = (possition[0] - camera_movement[0], possition[1] - camera_movement[1])
                    tracks[object][frame_num][track_id]['possition_adjusted'] = possition_adjusted