import numpy as np
import cv2
from typing import List, Tuple, Union

class ViewTransformer:
    """
    Lớp ViewTransformer dùng để biến đổi tọa độ từ không gian ảnh sang không gian thực tế của sân bóng đá.

    Lớp này thực hiện việc biến đổi phối cảnh (perspective transform) để chuyển đổi
    tọa độ pixel trong ảnh thành tọa độ thực tế trên sân bóng đá. Nó hữu ích trong
    việc phân tích vị trí và chuyển động của cầu thủ và bóng trên sân.

    Attributes:
        pixel_vertices (np.array): Tọa độ pixel của 4 góc sân trong ảnh.
        target_vertices (np.array): Tọa độ tương ứng của 4 góc sân trong không gian thực tế.
        persepctive_trasnformer (np.array): Ma trận biến đổi phối cảnh.

    Methods:
        transform_point(point): Biến đổi một điểm từ tọa độ ảnh sang tọa độ thực tế.
        add_transformed_position_to_tracks(tracks): Thêm vị trí đã biến đổi vào dữ liệu theo dõi.
    """

    def __init__(self):
        COURT_WIDTH = 68 # chiều rộng sân bóng
        COURT_LENGTH = 23.32

        # điểm góc của sân trong ảnh pixel
        self.pixel_vertices = np.array(
            [[110, 1035], 
            [265, 275], 
            [910, 260], 
            [1640, 915]]).astype(np.float32)

        # các điểm tương ứng trên sân thực tế
        self.target_vertices = np.array([ 
            [0, COURT_WIDTH],
            [0, 0],
            [COURT_LENGTH, 0],
            [COURT_LENGTH, COURT_WIDTH]
        ]).astype(np.float32)

        # biến đổi góc nhìn từ 4 điểm góc của hình ảnh pixel sang 4 điểm góc của hình ảnh thực tế. 
        self.persepctive_trasnformer = cv2.getPerspectiveTransform(self.pixel_vertices, self.target_vertices)

    def transform_point(self, point: Tuple[float, float]) -> Union[None, np.array]:
        """
        Hàm này thực hiện biến đổi 1 điểm trong không gian ảnh sang không gian thực tế sử dụng hàm cv2.perspectiveTransform.
        Kiểm tra nếu điểm đó nằm trong hình chữ nhật của sân bóng hay không. Nếu không, trả về None. Nếu có thực hiện biến đổi. 
        Args:
            point (Tuple[float, float]): Tọa độ của điểm trong không gian ảnh.
        Returns:
            None | np.array: None or Tọa độ của điểm đã biến đổi trong không gian thực tế.
        """
        point_copy = (int(point[0]), int(point[1]))
        # hàm sẽ kiểm tra xem điểm đó có nằm trong hình chữ nhật không, minDist = False để không cần thiết phải tính khoảng cách mà chỉ trả ra giá trị bool
        is_inside = cv2.pointPolygonTest(self.pixel_vertices, point_copy, False) >= 0 
        if not is_inside:
            return None

        reshaped_point = point.reshape(-1, 1, 2).astype(np.float32)
        tranform_point = cv2.perspectiveTransform(reshaped_point,self.persepctive_trasnformer)
        return tranform_point.reshape(-1, 2)


    def add_transformed_possition_to_tracks(self, tracks):
        """
        hàm này thực hiện biến đổi các điểm đã được chỉnh sửa "position_adjusted" của từng frame trong tracks từ không gian ảnh sang không gian thực tế.
        Args:
            tracks (dict): Dữ liệu theo dõi của các đối tượng trên sân bóng.
        Returns:
            None: Vị trí đã biến đổi được thêm vào tracks.
        """
        for object, object_tracks in tracks.items():
            for frame_num, track in enumerate(object_tracks):
                for track_id, track_info in track.items():
                    position = track_info['position_adjusted']
                    position = np.array(position)
                    position_trasnformed = self.transform_point(position)
                    if position_trasnformed is not None:
                        position_trasnformed = position_trasnformed.squeeze().tolist()
                    tracks[object][frame_num][track_id]['position_transformed'] = position_trasnformed
                    
