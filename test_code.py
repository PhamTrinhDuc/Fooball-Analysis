import numpy as np
from utils import read_video, save_video
from configs.config import APP_CFG
from source.tracker import Tracker
from source.team_assigner import TeamAssigner
from source.player_ball_assigner import PlayerBallAssigner
from source.camera_movement import CameraMovementEstimator
from utils import save_data


def main():
    video_frames = read_video(APP_CFG.input_video_path) # list[frame1, frame2, ...], 750 frames, array(1080, 1920, 3) frame
    tracker = Tracker()
    tracks = tracker.get_objects_track(video_frames)

    # Add possition to tracks
    tracker.add_possition_to_tracks(tracks)

    # # Draw camera movement
    camera_movement_estimator = CameraMovementEstimator(video_frames[0])
    camera_movement_per_frame = camera_movement_estimator.get_camera_movement(video_frames)
    
    # Interpolate Ball Positions
    tracks['ball'] = tracker.interpolate_ball_positions(tracks['ball'])

    # Assign Player Teams
    team_assigner = TeamAssigner()
    team_assigner.assign_team_color(video_frames[0], tracks['players'][0])
    
    for frame_number, player_track in enumerate(tracks['players']):
        for player_id, track in player_track.items():
            team = team_assigner.get_player_team(
                video_frames[frame_number],
                player_id,
                track['bbox']
            )

            tracks['players'][frame_number][player_id]['team'] = team
            tracks['players'][frame_number][player_id]['team_color'] = team_assigner.team_colors[team]

    # Assign Ball Aquisition
    player_ball_assigner = PlayerBallAssigner()
    team_ball_control = []
    for frame_num, player_track in enumerate(tracks['players']):
        ball_box = tracks['ball'][frame_num][1]['bbox']

        assigned_player = player_ball_assigner.assign_ball_to_player(player_track, ball_box)
        # print(f"Assigned player id: {assigned_player}")

        if assigned_player != -1:
            # print(tracks['players'][frame_num].keys())
            tracks['players'][frame_num][assigned_player]['has_ball'] = True

            team_ball_control.append(tracks['players'][frame_num][assigned_player]['team'])
        else:
            team_ball_control.append(team_ball_control[-1])
    
    team_ball_control = np.array(team_ball_control)

    # Draw Annotations
    output_video_frames = tracker.draw_annotations(video_frames, tracks, team_ball_control)

    print("=" * 50)
    output_video_frames = camera_movement_estimator.draw_camera_movement(output_video_frames, camera_movement_per_frame)
    print("=" * 50)
    save_video(output_video_frames, APP_CFG.output_video_path)

if __name__ == "__main__":
    main()


# from sklearn.cluster import KMeans
# import matplotlib.pyplot as plt
# import cv2

# image = cv2.imread('data/images/image.jpg')
# image_2d = image.reshape(-1, 3)
# kmeans = KMeans(n_clusters=2, random_state=0).fit(image_2d)

# labels = kmeans.labels_
# clustered_image = labels.reshape(image.shape[0], image.shape[1])
# # plt.imshow(clustered_image) (pixel 0 or 1)
# # plt.show()


# corner_cluster = clustered_image[0, 0], clustered_image[0, -1], clustered_image[-1, 0], clustered_image[-1, -1]
# print(len(kmeans.cluster_centers_[1]))