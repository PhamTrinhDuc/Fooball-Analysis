from sklearn.cluster import KMeans
import numpy as np
from typing import List, Tuple, Dict


class TeamAssigner:
    def __init__(self):
        """
        Lớp này được sử dụng để gán màu sắc và đội cho các cầu thủ trong một khung hình video.

        Attributes:
            team_colors (Dict): Lưu trữ màu sắc của mỗi đội.
            player_team_dict (Dict): Lưu trữ đội của mỗi cầu thủ.
    """
        self.team_colors = {}
        self.player_team_dict = {}


    def get_clustering_model(self, image: np.array) -> KMeans:
        
        """
        Tạo và huấn luyện một mô hình phân cụm KMeans trên một hình ảnh.

        Args:
            image: (np.array, shape=(H, W, 3)) image to cluster
        Returns:
            kmeans: (KMeans) clustering model
        """
        image_2d = image.reshape(-1, 3) # reshape the image to 2D array (H*W, 3)
        kmeans = KMeans(n_clusters=2, init='k-means++', n_init=1)
        kmeans.fit(image_2d)
        return kmeans
    
    def get_player_color(self, frame: np.array, bbox: List[float]) -> Tuple[float, float, float]:
        """
        Xác định màu sắc của cầu thủ dựa trên phần trên của bounding box.

        Args:
            frame: (np.array, shape=(H, W, 3)) frame containing the player
            bbox: (List[float], len(bbox) == 4) bounding box of the player
        Returns:
            player_color: (Tuple[int, int, int]) color of the player
        """
        image_top_half = frame[int(bbox[1]): int((bbox[1] + bbox[3]) / 2), int(bbox[0]): int((bbox[0] + bbox[2]) / 2)] # 
        kmeans = self.get_clustering_model(image_top_half)
        # get the labels for each pixel
        labels = kmeans.labels_
        # reshape labels to the shape of the image
        clustered_image = labels.reshape(image_top_half.shape[0], image_top_half.shape[1])
        # get corner pixel values
        corner_pixels = [clustered_image[0, 0], clustered_image[0, -1], clustered_image[-1, 0], clustered_image[-1, -1]]
        non_player_cluster = max(set(corner_pixels), key=corner_pixels.count)
        player_cluster = 1 - non_player_cluster 
        
        player_color = kmeans.cluster_centers_[player_cluster] # R G B

        return player_color
    
    def assign_team_color(self, frame: np.array, player_detections: Dict[str, List[float]]):
        """
        Gán màu sắc cho mỗi đội dựa trên màu sắc của tất cả các cầu thủ.

        Args:
            frame (np.array): Khung hình chứa các cầu thủ, có shape (H, W, 3).
            player_detections (Dict[str, List[float]]): Từ điển chứa thông tin bounding box của mỗi cầu thủ.

        Note:
            Phương thức này cập nhật thuộc tính team_colors và lưu mô hình kmeans.
        """
        
        player_colors = []

        for player_id, player_detection in player_detections.items():
            bbox = player_detection["bbox"]
            player_color = self.get_player_color(frame, bbox)
            player_colors.append(player_color)

        kmeans = KMeans(n_clusters=2, init='k-means++', n_init=10)
        kmeans.fit(player_colors)
        self.kmeans = kmeans

        self.team_colors[1] = kmeans.cluster_centers_[0]
        self.team_colors[2] = kmeans.cluster_centers_[1]
    
    def get_player_team(self, frame: np.array, player_id: int, player_bbox: List[float]) -> int:
        """
        Xác định đội của một cầu thủ dựa trên màu sắc của họ.

        Args:
            frame (np.array): Khung hình chứa cầu thủ, có shape (H, W, 3).
            player_id (int): ID của cầu thủ.
            player_bbox (List[float]): Bounding box của cầu thủ [x, y, width, height].

        Returns:
            int: ID của đội (1 hoặc 2).

        Note:
            Phương thức này cập nhật thuộc tính player_team_dict.
            Cầu thủ có ID 91 luôn được gán cho đội 1.
        """

        if player_id in self.player_team_dict:
            return self.player_team_dict[player_id]

        player_color = self.get_player_color(frame, player_bbox)

        team_id = self.kmeans.predict(player_color.reshape(1, -1))[0]
        team_id += 1

        if player_id == 100: # thu mon
            team_id = 1
        
        self.player_team_dict[player_id] = team_id

        return team_id