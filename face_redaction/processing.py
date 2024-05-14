import pathlib
from dataclasses import dataclass

import cv2
import face_recognition as rec
import numpy as np
from cv2.typing import MatLike

@dataclass
class FileInfo:
    name: str
    frame_width: int
    frame_height: int

from enum import Enum

class FaceDetectionModel(str, Enum):
    cnn = "cnn"
    default = "default"
    def __str__(self):
        return self.name
    
    
class FaceRedactionStrategy(str, Enum):
    blur = "blur"
    pixel = "pixel"
    solid = "solid"
    def __str__(self):
        return self.name


class MediaFileEditor:

    @property
    def image_formats_supported(self):
        return [".jpg", ".jpeg", ".png"]

    @property
    def video_formats_supported(self):
        return [".avi", ".mp4"]
    

    def is_valid_image(self, file_name:str) -> FileInfo:
        """
        Determines whether given file is valid and supported image
        """
        return pathlib.Path(file_name).suffix in self.image_formats_supported
        
    
    def is_valid_video(self, file_name:str) -> FileInfo:
        """
        Determines whether given file is valid and supported video
        """
        return pathlib.Path(file_name).suffix in self.video_formats_supported

    
    def redact_faces_in_image(
            self,
            input_file:str, 
            output_file:str, 
            detection_model: FaceDetectionModel,
            face_redaction_method: FaceRedactionStrategy,
            ) -> None:
        
        """
        Edits provided image, so that the people's faces are blurred. 
        Output is saved to a new file with the provided name.
        """

        # Load the input image, find all face locations in it
        image = cv2.imread(input_file) 
        face_locations = self._find_face_locations(image, detection_model=detection_model)

        # Redact faces in all locations using the method provided
        for top, right, bottom, left in face_locations:
            self._redact_face_roi(image, top, right, bottom, left, face_redaction_method=face_redaction_method)

        # Save the output image
        cv2.imwrite(output_file, image)

        # Close all OpenCV windows
        cv2.destroyAllWindows()


    def redact_faces_in_video(
            self,
            input_file:str, 
            output_file:str, 
            detection_model: FaceDetectionModel,
            face_redaction_method: FaceRedactionStrategy,
            video_codec:str = 'MP4V', #'XVID',
            frame_rate:float = 30.0,
            ) -> None:
        
        """
        Edits provided video, so that the people's faces are blurred. 
        Output is saved to a new file with the provided name.
        """

        # Get the video capture and info
        cap = cv2.VideoCapture(input_file)
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Define the codec and create output writer object
        # HINT: apparently ther's some issues with that: OpenCV: FFMPEG: tag 0x5634504d/'MP4V' is not supported with codec id 12 and format 'mp4 / MP4 (MPEG-4 Part 14)'
        #       see also: https://github.com/xingyizhou/CenterTrack/issues/40
        fourcc = cv2.VideoWriter_fourcc(*video_codec) 
        out = cv2.VideoWriter(output_file, fourcc, frame_rate, (frame_width, frame_height))

        # main processing loop
        while cap.isOpened():
            # Read next video frame
            ret, frame = cap.read()
            if not ret:
                break

            # Find all face locations in the frame
            face_locations = self._find_face_locations(frame, detection_model=detection_model)

            # Redact faces in all locations using the method provided
            for top, right, bottom, left in face_locations:
                self._redact_face_roi(frame, top, right, bottom, left, face_redaction_method=face_redaction_method)

            # Write the frame with blurred faces to the output video file
            out.write(frame)

        # Release the VideoCapture and VideoWriter objects
        cap.release()
        out.release()

        # Close all OpenCV windows
        cv2.destroyAllWindows()

    
    def _find_face_locations(
        self,
        image_or_frame: MatLike,
        detection_model: FaceDetectionModel,
    ) -> tuple[int, int, int, int]:
        """
        Find face locations
        """
        # Find all face locations in the frame
        # For different models available - see https://github.com/ageitgey/face_recognition/blob/master/examples/find_faces_in_picture_cnn.py
        face_locations = None
        if detection_model == FaceDetectionModel.cnn:
            face_locations = rec.face_locations(image_or_frame, number_of_times_to_upsample=0, model=str(detection_model))
        else:
            face_locations = rec.face_locations(image_or_frame)

        # Debug output (uncomment if needed)
        # if face_locations:
        #     print(f" *** Face locations: {len(face_locations)}, e.g. {str(face_locations[0])}")
        # else:
        #     print(" *** Face locations NOT FOUND")
        return face_locations
    

    def _redact_face_roi(
        self,
        image: MatLike,
        top: int,
        right: int,
        bottom: int,
        left: int,
        face_redaction_method: FaceRedactionStrategy,
    ) -> np.ndarray:
        """
        Redact face in provided region-of-interest (top, right, bottom, left) using given strategy
        """
        # HINT: consider using strategy (and correct parameters for gaussian / solid color retraction)
        blurred_face = None
        if face_redaction_method == FaceRedactionStrategy.blur:
            face_roi = image[top:bottom, left:right]
            kernel_size, filter_standard_deviation = (63,63), 100
            blurred_face = cv2.GaussianBlur(face_roi, kernel_size, filter_standard_deviation)
            image[top:bottom, left:right] = blurred_face
        elif face_redaction_method == FaceRedactionStrategy.pixel:
                face_roi = image[top:bottom, left:right]
                face_height, face_width = face_roi.shape[:2]

                # first resize face to a smaller size
                # then resize the small face back to the original size (will result in pixels)
                pixelation_factor = 0.05
                small_face_roi = cv2.resize(
                    face_roi, 
                    (int(face_width * pixelation_factor), int(face_height * pixelation_factor)), 
                    interpolation=cv2.INTER_NEAREST)
                pixelated_face = cv2.resize(small_face_roi, (face_width, face_height), interpolation=cv2.INTER_NEAREST)
                image[top:bottom, left:right] = pixelated_face
        else:
            default_color = (0, 0, 0)
            thickness = -1 # full thickness
            cv2.rectangle(image, (left, top), (right, bottom), default_color, thickness=thickness)
        return image

