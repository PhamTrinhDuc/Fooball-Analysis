from utils import get_center, measure_distance
from typing import List, Dict


class PlayerBallAssigner():
    def __init__(self):
        self.max_player_ball_distance = 70
    
    def assign_ball_to_player(self, players: Dict[int, Dict[str, List[float]]], ball_bbox: List[float]) -> int:
        ball_position = get_center(ball_bbox)

        MINIMUM_DISTANCE = 99999
        assigned_player=-1

        for player_id, player in players.items():
            player_bbox = player['bbox']

            distance_left = measure_distance((player_bbox[0],player_bbox[-1]),ball_position)
            distance_right = measure_distance((player_bbox[2],player_bbox[-1]),ball_position)
            distance = min(distance_left,distance_right)

            if distance < self.max_player_ball_distance:
                if distance < MINIMUM_DISTANCE:
                    MINIMUM_DISTANCE = distance
                    assigned_player = player_id

        return assigned_player