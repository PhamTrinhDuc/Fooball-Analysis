import yaml
from ultralytics import YOLO

class LoadConfig:
    def __init__(self) -> None:
        with open("./configs/config.yml") as cfg:
            app_config = yaml.load(cfg, Loader=yaml.FullLoader)
        
        self.load_directories(app_config=app_config)
    
    def load_directories(self, app_config):
        self.input_video_path = app_config["directories"]['input_video_path']
        self.output_video_path = app_config["directories"]['output_video_path']
        self.path_weight_yolo = app_config['directories']['path_weight_yolo']   
        self.frames_path = app_config['directories']['frames_path']
        self.camera_movement_path = app_config['directories']['camera_movement_path']
    
    def load_model(self):
        self.model = YOLO(self.path_weight_yolo)
        return self.model
    
APP_CFG = LoadConfig()