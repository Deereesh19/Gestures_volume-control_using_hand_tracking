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
STABILITY_WINDOW = 12
FRAME_WIDTH = 640
FRAME_HEIGHT = 480

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


def get_screen_size():
    if platform.system() == "Windows":
        try:
            user32 = __import__("ctypes").windll.user32
            return int(user32.GetSystemMetrics(0)), int(user32.GetSystemMetrics(1))
        except Exception:
            pass
    return 1600, 900


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


def save_dashboard(dashboard):
    screenshot_dir = Path(__file__).resolve().parent / "screenshots"
    screenshot_dir.mkdir(exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    dashboard_path = screenshot_dir / f"milestone4_dashboard_{timestamp}.png"
    cv2.imwrite(str(dashboard_path), dashboard)
    return dashboard_path


def process_hand_frame(hands, rgb_frame, runtime_state):
    if runtime_state["ready"]:
        return hands.process(rgb_frame)

    with suppress_native_stderr():
        results = hands.process(rgb_frame)

    runtime_state["ready"] = True
    return results


def classify_gesture(distance):
    if distance <= 55:
        return "Decrease Volume", (0, 130, 255)
    if distance >= 150:
        return "Increase Volume", (0, 210, 120)
    return "Hold Volume", (0, 220, 255)


def evaluate_gesture_quality(hand_landmarks, fingertip_history):
    thumb = hand_landmarks.landmark[4]
    index = hand_landmarks.landmark[8]

    margin = min(
        thumb.x,
        thumb.y,
        1.0 - thumb.x,
        1.0 - thumb.y,
        index.x,
        index.y,
        1.0 - index.x,
        1.0 - index.y,
    )
    margin_score = float(np.clip(margin / 0.12, 0.0, 1.0))
    distance_score = float(
        np.clip(
            (math.hypot(index.x - thumb.x, index.y - thumb.y) - 0.03) / 0.18,
            0.0,
            1.0,
        )
    )

    stability_score = 0.5
    if len(fingertip_history) >= 4:
        points = np.array(fingertip_history, dtype=np.float32)
        jitter = float(np.mean(np.std(points, axis=0)))
        stability_score = float(np.clip(1.0 - (jitter / 25.0), 0.0, 1.0))

    score = int(round((margin_score * 0.4 + distance_score * 0.25 + stability_score * 0.35) * 100))

    if score >= 80:
        return score, "Excellent", (0, 210, 120)
    if score >= 60:
        return score, "Good", (0, 220, 255)
    if score >= 40:
        return score, "Fair", (0, 160, 255)
    return score, "Poor", (0, 90, 255)


def build_mini_graph(history, width=330, height=170):
    graph = np.full((height, width, 3), 24, dtype=np.uint8)

    top_pad = max(28, int(height * 0.18))
    bottom_pad = max(34, int(height * 0.16))
    left_pad = max(44, int(width * 0.11))
    right_pad = max(14, int(width * 0.04))

    x0, y0 = left_pad, height - bottom_pad
    x1, y1 = width - right_pad, top_pad

    cv2.rectangle(graph, (x0, y1), (x1, y0), (38, 38, 38), -1)
    cv2.rectangle(graph, (x0, y1), (x1, y0), (85, 85, 85), 1)

    for level in range(0, 101, 25):
        y = int(np.interp(level, [0, 100], [y0, y1]))
        cv2.line(graph, (x0, y), (x1, y), (55, 55, 55), 1)
        cv2.putText(graph, str(level), (10, y + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (180, 180, 180), 1)

    if len(history) > 1:
        points = []
        total = max(1, len(history) - 1)
        for index, (_, volume_percent) in enumerate(history):
            x = int(np.interp(index, [0, total], [x0, x1]))
            y = int(np.interp(volume_percent, [0, 100], [y0, y1]))
            points.append((x, y))

        for index in range(1, len(points)):
            intensity = int(np.interp(index, [1, len(points) - 1], [120, 255])) if len(points) > 2 else 220
            cv2.line(graph, points[index - 1], points[index], (0, intensity, 180), 2)

        cv2.circle(graph, points[-1], 5, (0, 210, 255), -1)

    title_x = max(18, width // 2 - 82)
    cv2.putText(graph, "Volume Trend", (title_x, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (235, 235, 235), 2)
    cv2.putText(graph, "time", (width - 54, height - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (170, 170, 170), 1)
    return graph


def draw_volume_bar(canvas, x, y, height, volume_percent):
    cv2.rectangle(canvas, (x, y), (x + 74, y + height), (70, 70, 70), 2)
    fill_top = int(np.interp(volume_percent, [0, 100], [y + height, y]))
    cv2.rectangle(canvas, (x, fill_top), (x + 74, y + height), (0, 210, 120), cv2.FILLED)
    cv2.putText(canvas, "Volume", (x - 2, y - 16), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (245, 245, 245), 1)


def compose_dashboard(
    frame,
    volume_percent,
    distance,
    gesture_label,
    gesture_color,
    quality_score,
    quality_label,
    quality_color,
    hand_detected,
    graph_history,
    info_message,
    dashboard_size,
):
    dashboard_width, dashboard_height = dashboard_size
    dashboard = np.full((dashboard_height, dashboard_width, 3), (16, 16, 16), dtype=np.uint8)

    outer_margin = max(18, dashboard_width // 60)
    top_margin = max(28, dashboard_height // 34)
    header_y = top_margin + 10
    footer_height = max(52, dashboard_height // 14)
    gap = max(26, dashboard_width // 44)
    right_panel_width = max(430, int(dashboard_width * 0.28))
    left_panel_width = dashboard_width - (outer_margin * 2) - gap - right_panel_width
    content_top = top_margin + 38
    content_bottom = dashboard_height - outer_margin - footer_height - 16
    content_height = content_bottom - content_top
    right_x = outer_margin + left_panel_width + gap

    dashboard[:, : right_x - gap // 2] = (22, 22, 22)
    dashboard[:, right_x - gap // 2 :] = (30, 30, 30)

    feed_width = left_panel_width - 16
    feed_height = content_height - 16
    feed = cv2.resize(frame, (feed_width, feed_height))
    feed_x = outer_margin + 8
    feed_y = content_top + 8
    dashboard[feed_y:feed_y + feed_height, feed_x:feed_x + feed_width] = feed
    cv2.rectangle(
        dashboard,
        (outer_margin, content_top),
        (outer_margin + left_panel_width, content_top + content_height),
        (80, 80, 80),
        2,
    )
    cv2.rectangle(dashboard, (right_x, content_top), (dashboard_width - outer_margin, content_bottom), (45, 45, 45), 2)

    cv2.putText(dashboard, "Milestone 4 - Final UI and Feedback", (outer_margin + 10, header_y), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (245, 245, 245), 2)
    cv2.putText(dashboard, "Gesture Volume Controller", (right_x + 28, content_top + 56), cv2.FONT_HERSHEY_SIMPLEX, 0.95, (255, 255, 255), 2)

    status_text = "Tracking Active" if hand_detected else "Waiting for Hand"
    status_color = (0, 210, 120) if hand_detected else (0, 100, 255)
    cv2.putText(dashboard, status_text, (right_x + 28, content_top + 104), cv2.FONT_HERSHEY_SIMPLEX, 0.84, status_color, 2)

    volume_y = content_top + 142
    volume_bar_height = min(210, max(165, int(content_height * 0.19)))
    draw_volume_bar(dashboard, right_x + 36, volume_y, volume_bar_height, volume_percent)
    cv2.putText(dashboard, f"{int(volume_percent)}%", (right_x + 150, volume_y + 82), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 220, 120), 3)
    cv2.putText(dashboard, f"Distance: {int(distance)} px", (right_x + 150, volume_y + 132), cv2.FONT_HERSHEY_SIMPLEX, 0.74, (230, 230, 230), 2)

    card_x0 = right_x + 24
    card_x1 = dashboard_width - outer_margin - 24
    card_width = card_x1 - card_x0
    card_height = 86
    gesture_card_y = volume_y + volume_bar_height + 16
    quality_card_y = gesture_card_y + card_height + 14
    graph_y = quality_card_y + card_height + 20

    cv2.rectangle(dashboard, (card_x0, gesture_card_y), (card_x1, gesture_card_y + card_height), (36, 36, 36), -1)
    cv2.putText(dashboard, "Gesture Type", (card_x0 + 18, gesture_card_y + 34), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (180, 180, 180), 1)
    cv2.putText(dashboard, gesture_label, (card_x0 + 18, gesture_card_y + 70), cv2.FONT_HERSHEY_SIMPLEX, 0.86, gesture_color, 2)

    cv2.rectangle(dashboard, (card_x0, quality_card_y), (card_x1, quality_card_y + card_height), (36, 36, 36), -1)
    cv2.putText(dashboard, "Gesture Quality", (card_x0 + 18, quality_card_y + 34), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (180, 180, 180), 1)
    cv2.putText(dashboard, f"{quality_label} ({quality_score}%)", (card_x0 + 18, quality_card_y + 70), cv2.FONT_HERSHEY_SIMPLEX, 0.86, quality_color, 2)

    available_graph_height = content_bottom - graph_y - 18
    graph_height = max(120, available_graph_height)
    graph_width = card_width
    graph = build_mini_graph(graph_history, width=graph_width, height=graph_height)
    dashboard[graph_y:graph_y + graph_height, card_x0:card_x0 + graph_width] = graph

    footer = info_message or "Keys: q = quit, s = save dashboard screenshot"
    footer_top = dashboard_height - outer_margin - footer_height
    cv2.rectangle(dashboard, (outer_margin, footer_top), (dashboard_width - outer_margin, footer_top + footer_height), (24, 24, 24), -1)
    cv2.putText(dashboard, footer, (outer_margin + 18, footer_top + int(footer_height * 0.68)), cv2.FONT_HERSHEY_SIMPLEX, 0.82, (210, 210, 210), 2)
    return dashboard


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
    current_gesture = "Waiting for Hand"
    gesture_color = (0, 100, 255)
    quality_score = 0
    quality_label = "Poor"
    quality_color = (0, 90, 255)
    graph_history = deque(maxlen=GRAPH_HISTORY_LENGTH)
    fingertip_history = deque(maxlen=STABILITY_WINDOW)
    info_message = ""
    info_message_frames = 0
    runtime_state = {"ready": False}
    screen_width, screen_height = get_screen_size()
    dashboard_size = (
        max(1280, min(1720, screen_width - 40)),
        max(820, min(980, screen_height - 90)),
    )

    cap = open_camera()
    if cap is None:
        print("Error: Camera not accessible")
        hands.close()
        raise SystemExit(1)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    failed_reads = 0
    max_failed_reads = 30

    try:
        cv2.namedWindow("Milestone 4 - Final Application", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Milestone 4 - Final Application", dashboard_size[0], dashboard_size[1])

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
            results = process_hand_frame(hands, rgb, runtime_state)

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

                midpoint = ((x1 + x2) // 2, (y1 + y2) // 2)
                cv2.circle(frame, midpoint, 7, (0, 210, 255), cv2.FILLED)

                current_distance = math.hypot(x2 - x1, y2 - y1)
                target_percent = map_distance_to_percent(current_distance)
                current_gesture, gesture_color = classify_gesture(current_distance)

                fingertip_history.append((x1, y1, x2, y2))
                quality_score, quality_label, quality_color = evaluate_gesture_quality(hand_landmarks, fingertip_history)
            else:
                current_gesture = "Waiting for Hand"
                gesture_color = (0, 100, 255)
                quality_score = 0
                quality_label = "Poor"
                quality_color = (0, 90, 255)
                fingertip_history.clear()

            smooth_percent = smooth_value(smooth_percent, target_percent)
            smooth_percent = float(np.clip(smooth_percent, 0.0, 100.0))
            volume_controller.SetMasterVolumeLevelScalar(smooth_percent / 100.0, None)
            graph_history.append((current_distance, smooth_percent))

            if info_message_frames > 0:
                info_message_frames -= 1
            else:
                info_message = ""

            dashboard = compose_dashboard(
                frame=frame,
                volume_percent=smooth_percent,
                distance=current_distance,
                gesture_label=current_gesture,
                gesture_color=gesture_color,
                quality_score=quality_score,
                quality_label=quality_label,
                quality_color=quality_color,
                hand_detected=hand_detected,
                graph_history=graph_history,
                info_message=info_message,
                dashboard_size=dashboard_size,
            )

            cv2.imshow("Milestone 4 - Final Application", dashboard)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord("s"):
                saved_path = save_dashboard(dashboard)
                info_message = f"Saved screenshot: {saved_path.name}"
                info_message_frames = 90
    finally:
        hands.close()
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
