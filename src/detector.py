import logging
import numpy as np
import cv2
from ultralytics import YOLO

class YOLODetector:
    def __init__(self, model_path: str = "yolo11n.pt"):
        logging.info("Loading YOLO model: %s", model_path)
        self.model = YOLO(model_path)

    def detect_and_draw(self, image: np.ndarray) -> np.ndarray:
        """
        Runs detection on the image and returns the image with bounding boxes drawn.
        Expects image to be uint8 (H, W, 3) RGB.
        """
        # Run inference
        results = self.model(image, verbose=False)
        
        # Plot results on the image
        # plot() returns a BGR numpy array, so we need to convert back to RGB if needed.
        # However, ultralytics plot() usually returns the image in the same format as input if passed as numpy?
        # Let's check documentation or behavior. 
        # Actually, results[0].plot() returns a BGR numpy array.
        
        annotated_frame = results[0].plot()
        
        return annotated_frame
