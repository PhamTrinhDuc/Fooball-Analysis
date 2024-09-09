from typing import List, Tuple

def get_center(bbox: List[float]) -> Tuple[int, int]:
    x1, y1, x2, y2 = bbox # góc trái trên và góc phải dưới
    x_center = int((x1 + x2) / 2)
    y_center = int((y1 + y2)/ 2)
    return x_center, y_center

def get_bbox_width(bbox: List[float]) -> int:
    x1, y1, x2, y2 = bbox # góc trái trên và góc phải dưới
    return x2 - x1

def get_position_foot(bbox: List[float]) -> Tuple[int, int]:
    x1, y1, x2, y2 = bbox
    return int((x1 + x2) / 2), int(y2)

def measure_distance(p1: Tuple[float, float], p2: Tuple[float, float]) -> float:
    return ((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)**0.5

def measure_xy_distance(p1: Tuple[float, float], p2: Tuple[float, float]) -> Tuple[float, float]:
    return (p1[0] - p2[0], p1[1] - p2[1])