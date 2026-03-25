# Milestone 4: Gesture Volume Control Web App

This milestone converts the hand-gesture volume control project into a Flask-based web application. It uses a webcam to detect the distance between the thumb tip and index finger tip, maps that distance to system volume, and shows live visual feedback in the browser.

## Features

- Live webcam stream inside a Flask web interface
- Hand detection using MediaPipe Hands
- Gesture-based volume mapping using thumb-index distance
- Real-time system status panel
- Live volume percentage bar
- Volume trend graph in the dashboard
- Start and Stop camera controls
- Placeholder screen when the camera is inactive
- Windows system volume control through `pycaw`

## Project Structure

```text
Milestone4/
├── app.py
├── templates/
│   └── index.html
├── static/
│   └── style.css
├── requirements.txt
└── README.md
```

## Technologies Used

- Python
- Flask
- OpenCV
- MediaPipe
- NumPy
- Pycaw
- Comtypes
- HTML, CSS, JavaScript

## How It Works

1. The Flask server starts a webcam stream.
2. Frames are captured in a background thread for smoother browser playback.
3. MediaPipe detects one hand and extracts landmarks.
4. The distance between thumb tip and index finger tip is measured.
5. That distance is converted into a volume percentage.
6. On Windows, the system volume is updated using `pycaw`.
7. The browser dashboard shows:
   - webcam stream
   - camera status
   - gesture status
   - gesture quality
   - current volume
   - volume trend graph

## Gesture Logic

The app uses thumb-index distance in pixels:

- `0-30 px`: Pinch
- `30-60 px`: Close
- `60-110 px`: Medium
- `110-200 px`: Far

The measured distance is also mapped continuously to a `0-100%` volume range.

## Requirements

- Python 3.11 recommended
- Windows OS for system volume control
- Working webcam
- Virtual environment recommended

## Installation

From the `Milestone4` folder:

```powershell
python -m venv venv
venv\Scripts\activate
python -m pip install -r requirements.txt
```

## Run the Application

```powershell
python app.py
```

Then open:

```text
http://127.0.0.1:5000
```

## Controls

- `Start`: starts webcam capture and live gesture processing
- `Stop`: stops webcam capture and shows the inactive placeholder screen

## API Endpoints

- `/` : main dashboard page
- `/video_feed` : MJPEG webcam stream
- `/placeholder_frame` : inactive placeholder image
- `/start_camera` : starts camera processing
- `/stop_camera` : stops camera processing
- `/status` : returns current gesture, volume, and camera state as JSON

## Current Optimizations

- Background camera-processing thread
- Reduced MediaPipe processing resolution for better responsiveness
- Camera buffer size reduced to lower lag
- Lightweight MediaPipe hand model
- Browser graph updates from status polling

## Limitations

- System volume control is supported only on Windows
- Browser MJPEG streaming is heavier than native desktop rendering
- Low lighting or poor camera angle may reduce hand detection quality
- If another app is using the webcam, the stream may not start

## Future Improvements

- Better gesture classification labels
- Smoothed graph rendering
- Fullscreen dashboard mode
- More robust multi-gesture support
- More efficient streaming method than MJPEG

## Author

Milestone 4 implementation for the Gesture Volume Control project.
