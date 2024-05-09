# Face-redaction

Face-redaction is a command-line (CLI) aplication to redact faces in images and videos.
Face-redaction utilizes AI models to detect faces in provided images and video files, and replace them with either a blurred, or solid-color rectangles.

## How it works

The face redaction process involves two steps:
1. detecting face coordinates (also known as: region of interests).
   `face-recognition` library is used for face detection. It allows detecting faces using different models, such as:
   * `hog` - default model, faster
   * `cnn` - NN model, considerably slower (much better performance with CUDA)
2. replace the found coordinates with given redaction method
   For image processing the `opencv-python` library
 - `blur` - uses `cv2.GaussianBlur`
 - `solid` - just a solid-color rectangles

## Installation
```bash
poetry install
```

## Launch
general help:
```bash
poetry run redact --help
```

basic information about the tool and supported image and video formats:
```bash
poetry run redact info
```

redact faces in a video:
```bash
poetry run redact redact-faces MY_VIDEO_FILE.mp4
```

redact faces using convolution neural network model, with solid color replacement
```bash
poetry run redact redact-faces MY_VIDEO_FILE.mp4
```

### Face detection options
- default - `face-recognition` `hog` model
- cnn - convolution neural network


### Face redaction options
- blur - blurred face
- solid - face replaced with a solid color rectangle