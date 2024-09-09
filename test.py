# import cv2
# import numpy as np

# # Đọc ảnh
# image = cv2.imread('source/camera_movement/doc.png')
# gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# # Các tham số
# max_corners = 100  # Số điểm đặc trưng tối đa
# quality_level = 0.01  # Chất lượng của điểm đặc trưng
# min_distance = 10  # Khoảng cách tối thiểu giữa các điểm đặc trưng

# # Phát hiện các điểm đặc trưng
# corners = cv2.goodFeaturesToTrack(gray, maxCorners=max_corners, qualityLevel=quality_level, minDistance=min_distance)
# # shape (100, 1, 2)

# # # Vẽ các điểm đặc trưng lên ảnh
# if corners is not None:
#     for corner in corners:
#         print(corner.ravel())
#         x, y = corner.ravel()
#         print(type(x))
#         break

        # cv2.circle(image, (x, y), 3, 255, -1)  # Vẽ hình tròn tại mỗi điểm đặc trưng

# # Hiển thị ảnh kết quả
# cv2.imshow('Good Features to Track', image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

import numpy as np

res = np.array([[110, 1035], 
        [265, 275], 
        [910, 260], 
        [1640, 915]])

print(res.shape)