# Face Anonymizer

A real-time face detection and anonymization application using OpenCV and Python.

## Features

- Real-time face detection using Haar cascades
- Face anonymization with Gaussian blur
- Live FPS counter
- Simple and efficient implementation

## Requirements

- Python 3.6+
- OpenCV (cv2)

## Installation

1. Clone this repository
2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

## Usage

Run the application:
```
python main.py
```

- Press 'q' to quit the application
- The application will automatically detect and blur faces in real-time

## Configuration

You can modify the following constants in `main.py`:
- `BLUR_KERNEL_SIZE`: Size of the blur kernel
- `BLUR_SIGMA`: Blur intensity
- `SCALE_FACTOR`: Face detection scale factor
- `MIN_NEIGHBORS`: Minimum neighbors for face detection