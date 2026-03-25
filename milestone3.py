import math
import os
import platform
import warnings
from collections import deque
from contextlib import contextmanager
from ctypes import POINTER, cast
from datetime import datetime
from pathlib import Path

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["GLOG_minloglevel"] = "2"
warnings.filterwarnings(
    "ignore",
    message=r"SymbolDatabase\.GetPrototype\(\) is deprecated\..*",
    category=UserWarning,
)

import cv2
import mediapipe as mp
import numpy as np


MIN_DISTANCE = 30.0
MAX_DISTANCE = 250.0
SMOOTHING_FACTOR = 0.2
GRAPH_HISTORY_LENGTH = 90

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils


@contextmanager
def suppress_native_stderr():
    saved_stderr = os.dup(2)
    try:
        with open(os.devnull, "w", encoding="utf-8") as devnull:
            os.dup2(devnull.fileno(), 2)
            yield
    finally:
        os.dup2(saved_stderr, 2)
        os.close(saved_stderr)


def open_camera():
    for backend in (cv2.CAP_DSHOW, cv2.CAP_MSMF, cv2.CAP_ANY):
        for cam_index in (0, 1, 2):
            cap = cv2.VideoCapture(cam_index, backend)
            if not cap.isOpened():
                cap.release()
                continue

            ok, _ = cap.read()
            if ok:
                return cap
            cap.release()
    return None


def get_volume_controller():
    if platform.system() != "Windows":
        raise RuntimeError("System volume control is supported only on Windows in this project.")

    try:
        from comtypes import CLSCTX_ALL
        from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
    except Exception as exc:
        raise RuntimeError(f"Required audio libraries are not available: {exc}") from exc

    devices = AudioUtilities.GetSpeakers()
    interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
    return cast(interface, POINTER(IAudioEndpointVolume))


def map_distance_to_percent(distance):
    clipped = float(np.clip(distance, MIN_DISTANCE, MAX_DISTANCE))
    return float(np.interp(clipped, [MIN_DISTANCE, MAX_DISTANCE], [0, 100]))


def smooth_value(current, target, factor=SMOOTHING_FACTOR):
    return current + (target - current) * factor


def save_screenshots(frame, graph):
    screenshot_dir = Path(__file__).resolve().parent / "screenshots"
    screenshot_dir.mkdir(exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    frame_path = screenshot_dir / f"volume_display_{timestamp}.png"
    graph_path = screenshot_dir / f"distance_graph_{timestamp}.png"
    combined_path = screenshot_dir / f"gesture_volume_output_{timestamp}.png"

    graph_preview = cv2.resize(graph, (frame.shape[1], frame.shape[0]))
    combined = np.hstack((frame, graph_preview))

    cv2.imwrite(str(frame_path), frame)
    cv2.imwrite(str(graph_path), graph)
    cv2.imwrite(str(combined_path), combined)

    return frame_path, graph_path, combined_path


def build_mapping_graph(distance, volume_percent, history):
    graph = np.full((420, 620, 3), 245, dtype=np.uint8)

    x0, y0 = 70, 360
    x1, y1 = 570, 50

    cv2.rectangle(graph, (x0, y1), (x1, y0), (210, 210, 210), -1)
    cv2.rectangle(graph, (x0, y1), (x1, y0), (80, 80, 80), 2)

    for i in range(0, 101, 10):
        y = int(np.interp(i, [0, 100], [y0, y1]))
        cv2.line(graph, (x0, y), (x1, y), (225, 225, 225), 1)
        cv2.putText(graph, str(i), (30, y + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (70, 70, 70), 1)

    for d in range(int(MIN_DISTANCE), int(MAX_DISTANCE) + 1, 20):
        x = int(np.interp(d, [MIN_DISTANCE, MAX_DISTANCE], [x0, x1]))
        cv2.line(graph, (x, y0), (x, y1), (225, 225, 225), 1)
        cv2.putText(graph, str(d), (x - 13, y0 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (70, 70, 70), 1)

    cv2.line(graph, (x0, y0), (x1, y1), (0, 150, 140), 2)
    if len(history) > 1:
        points = []
        for hist_distance, hist_volume in history:
            plot_distance = float(np.clip(hist_distance, MIN_DISTANCE, MAX_DISTANCE))
            plot_volume = float(np.clip(hist_volume, 0.0, 100.0))
            plot_x = int(np.interp(plot_distance, [MIN_DISTANCE, MAX_DISTANCE], [x0, x1]))
            plot_y = int(np.interp(plot_volume, [0, 100], [y0, y1]))
            points.append((plot_x, plot_y))

        history_span = max(1, len(points) - 1)
        for index in range(1, len(points)):
            intensity = int(120 + (135 * index / history_span))
            cv2.line(graph, points[index - 1], points[index], (255 - intensity // 3, intensity, 80), 2)

    cur_x = int(np.interp(distance, [MIN_DISTANCE, MAX_DISTANCE], [x0, x1]))
    cur_y = int(np.interp(volume_percent, [0, 100], [y0, y1]))
    cv2.circle(graph, (cur_x, cur_y), 7, (0, 90, 255), -1)
    cv2.circle(graph, (cur_x, cur_y), 9, (255, 255, 255), 2)
    cv2.putText(graph, "Distance to Volume Mapping", (170, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (40, 40, 40), 2)
    cv2.putText(
        graph,
        f"Current: {int(distance)} px -> {int(volume_percent)}%",
        (155, 52),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (30, 30, 30),
        1,
    )
    cv2.putText(graph, "Distance (pixels)", (240, 395), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (50, 50, 50), 1)
    cv2.putText(graph, "Volume %", (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (50, 50, 50), 1)
    return graph


def process_hand_frame(hands, rgb_frame, suppress_warmup_state):
    if suppress_warmup_state["ready"]:
        return hands.process(rgb_frame)

    with suppress_native_stderr():
        results = hands.process(rgb_frame)

    suppress_warmup_state["ready"] = True
    return results


def draw_volume_panel(frame, volume_percent, distance, detected, info_message=""):
    cv2.rectangle(frame, (35, 120), (90, 390), (0, 220, 120), 2)
    bar_top = int(np.interp(volume_percent, [0, 100], [390, 120]))
    cv2.rectangle(frame, (35, bar_top), (90, 390), (0, 220, 120), cv2.FILLED)

    cv2.rectangle(frame, (130, 20), (460, 130), (30, 30, 30), -1)
    cv2.putText(frame, "Current Volume", (150, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (220, 220, 220), 2)
    cv2.putText(frame, f"{int(volume_percent)}%", (225, 102), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 220, 120), 3)
    cv2.putText(frame, f"Distance: {int(distance)} px", (180, 155), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 240, 240), 2)
    status = "Gesture: Active" if detected else "Gesture: Not detected"
    color = (0, 220, 120) if detected else (0, 100, 255)
    cv2.putText(frame, status, (180, 185), cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2)
    cv2.putText(frame, "Keys: q=quit  s=save screenshots", (12, 470), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
    if info_message:
        cv2.rectangle(frame, (120, 430), (620, 462), (20, 20, 20), -1)
        cv2.putText(frame, info_message, (130, 452), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (120, 255, 180), 1)


def main():
    with suppress_native_stderr():
        hands = mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7,
        )

    try:
        volume_controller = get_volume_controller()
    except Exception as exc:
        print(f"Error: Could not initialize system volume controller. {exc}")
        hands.close()
        raise SystemExit(1)

    smooth_percent = volume_controller.GetMasterVolumeLevelScalar() * 100.0
    current_distance = MIN_DISTANCE
    graph_history = deque(maxlen=GRAPH_HISTORY_LENGTH)
    info_message = ""
    info_message_frames = 0
    mediapipe_runtime = {"ready": False}

    cap = open_camera()
    if cap is None:
        print("Error: Camera not accessible")
        hands.close()
        raise SystemExit(1)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    failed_reads = 0
    max_failed_reads = 30

    try:
        while True:
            success, frame = cap.read()
            if not success:
                failed_reads += 1
                if failed_reads >= max_failed_reads:
                    print("Failed to grab frame. Check if another app is using the camera.")
                    break
                continue

            failed_reads = 0
            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = process_hand_frame(hands, rgb, mediapipe_runtime)
            hand_detected = False
            target_percent = smooth_percent

            if results.multi_hand_landmarks:
                hand_detected = True
                hand_landmarks = results.multi_hand_landmarks[0]
                mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                h, w, _ = frame.shape
                x1 = int(hand_landmarks.landmark[4].x * w)
                y1 = int(hand_landmarks.landmark[4].y * h)
                x2 = int(hand_landmarks.landmark[8].x * w)
                y2 = int(hand_landmarks.landmark[8].y * h)

                cv2.circle(frame, (x1, y1), 9, (255, 80, 0), cv2.FILLED)
                cv2.circle(frame, (x2, y2), 9, (255, 80, 0), cv2.FILLED)
                cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 160), 3)

                current_distance = math.hypot(x2 - x1, y2 - y1)
                target_percent = map_distance_to_percent(current_distance)

            smooth_percent = smooth_value(smooth_percent, target_percent)
            smooth_percent = float(np.clip(smooth_percent, 0.0, 100.0))
            volume_controller.SetMasterVolumeLevelScalar(smooth_percent / 100.0, None)
            graph_history.append((current_distance, smooth_percent))

            if info_message_frames > 0:
                info_message_frames -= 1
            else:
                info_message = ""

            draw_volume_panel(frame, smooth_percent, current_distance, hand_detected, info_message)
            graph = build_mapping_graph(current_distance, smooth_percent, graph_history)

            cv2.imshow("Milestone 3 - Real-time Volume Display", frame)
            cv2.imshow("Milestone 3 - Distance to Volume Graph", graph)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord("s"):
                _, _, combined_path = save_screenshots(frame, graph)
                info_message = f"Saved screenshot: {combined_path.name}"
                info_message_frames = 90
    finally:
        hands.close()
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
