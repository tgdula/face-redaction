from typing import Optional
import mediapipe as mp
import cv2
from cv2.typing import MatLike

def find_face_locations(
        image_or_frame: MatLike,
    ) -> tuple[int, int, int, int]:
        """
        Find face locations using `mediapipe` library
        see: https://developers.google.com/mediapipe/solutions/vision/face_detector

        HINT: this library is not ideal, e.g.:
              - detected faces in photos of hands etc
              - Warnings: W0000 00:00:1715935444.027180  162064 inference_feedback_manager.cc:114] Feedback manager requires a model with a single signature inference. Disabling support for feedback tensors.
        """
        def detect_face_roi(image_or_frame: MatLike,detection):
            image_height, image_width, _ = image_or_frame.shape
            bboxC = detection.location_data.relative_bounding_box
            x, y, w, h = (
                int(bboxC.xmin * image_width), 
                int(bboxC.ymin * image_height), 
                int(bboxC.width * image_width), 
                int(bboxC.height * image_height))
            return (y, x+w, y+h, x)
             
        mp_face_detection = mp.solutions.face_detection
        image_rgb = cv2.cvtColor(image_or_frame, cv2.COLOR_BGR2RGB) # HINT: this conversion is crucial for Mediapipe

        face_locations = []
        with mp_face_detection.FaceDetection(min_detection_confidence=0.8) as face_detection:
            results = face_detection.process(image_rgb)
            if results.detections:
                face_locations = [detect_face_roi(image_or_frame, detection) for detection in results.detections]

        return face_locations