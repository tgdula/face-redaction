# Face-redaction

Face-redaction is a command-line (CLI) aplication to redact faces in images and videos.
Face-redaction utilizes AI models to detect faces in provided images and video files, and replace them with either a blurred, or solid-color rectangles.

## How it works

The face redaction process involves two steps:
1. detecting face coordinates (also known as: region of interests).
   `face-recognition` library is used for face detection. It allows detecting faces using different models, such as:
   * `hog` - histogram of oriented gradents model - default & faster
   * `cnn` - convolutional NN model, considerably slower (much better performance with CUDA)
   See [face detection in towardsdatascience.com](https://towardsdatascience.com/cnn-based-face-detector-from-dlib-c3696195e01c)
2. replace the found coordinates with given redaction method: blur, pixel, solid color
   For image processing the `opencv-python` library is used


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

redact faces using convolutional neural network model, with solid color replacement
```bash
poetry run redact redact-faces --face-detection-model=cnn --face-redaction-method=solid MY_VIDEO_FILE.mp4
```

### Face detection options
 - `default` - `face-recognition` `hog` model
 - `cnn` - convolution neural network


### Face redaction options
 - `blur` - uses `cv2.GaussianBlur`
 - `pixel` - replace face rectangle with pixels
 - `solid` - just a solid-color rectangles


## Troubleshooting
### `dlib` installation with GPU support
`Face-redaction` uses the `dlib` library, which seems challenging to install with GPU support. I recommend after [https://ankitmishra723.medium.com/dlib-setup-for-windows-python-0d9ea92a3e18](https://ankitmishra723.medium.com/dlib-setup-for-windows-python-0d9ea92a3e18) to install some the pre-required libraries with `pip` and the `dlib` library with conda:
```
pip install build
pip install cmake
conda install -c conda-forge dlib
```

To check whether the `dlib` actually uses GPU