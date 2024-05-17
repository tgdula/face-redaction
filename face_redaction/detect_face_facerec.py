from typing import Optional
import face_recognition as rec
from cv2.typing import MatLike

def find_face_locations(
        image_or_frame: MatLike,
        detection_model: Optional[str] = None,
    ) -> tuple[int, int, int, int]:
        """
        Find face locations using `face-recognition` library (internally: dlib)
        """
        face_locations = (
             rec.face_locations(image_or_frame, number_of_times_to_upsample=0, model=detection_model) if detection_model
             else rec.face_locations(image_or_frame)
        )
        return face_locations