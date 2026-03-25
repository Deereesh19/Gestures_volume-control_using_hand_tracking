# Import OpenCV library for camera and image processing
import cv2

# Import MediaPipe
import mediapipe as mp

# Import drawing utilities to draw hand landmarks
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

# Import math for distance calculation
import math


def classify_gesture(distance):
    # Thresholds tuned for a 640x480 frame.
    if distance < 45:
        return "Gesture: Pinch (Decrease)", (0, 120, 255)
    if distance > 130:
        return "Gesture: Open (Increase)", (0, 220, 120)
    return "Gesture: Neutral (Hold)", (255, 220, 0)


def main():
    # ===================== STEP 1: HAND DETECTION SETUP =====================
    hands = mp_hands.Hands(
        static_image_mode=False,          # False means continuous video detection
        max_num_hands=1,                  # Detect only one hand
        model_complexity=0,               # Faster model (0 = lightweight)
        min_detection_confidence=0.6,     # Minimum confidence for detection
        min_tracking_confidence=0.6       # Minimum confidence for tracking
    )

    # Start webcam (0 means default camera)
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not access webcam.")
        hands.close()
        raise SystemExit(1)

    # Set camera width
    cap.set(3, 640)

    # Set camera height
    cap.set(4, 480)

    failed_reads = 0
    max_failed_reads = 30

    try:
        # Infinite loop to read camera frames
        while True:
            # Read frame from camera
            success, frame = cap.read()

            # If frame is not captured properly, exit loop
            if not success:
                failed_reads += 1
                if failed_reads >= max_failed_reads:
                    print("Warning: Failed to read frame from webcam repeatedly.")
                    break
                continue

            failed_reads = 0

            # Convert BGR image to RGB (MediaPipe requires RGB format)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Process the frame to detect hands
            result = hands.process(rgb)

            gesture_text = "Gesture: Not detected"
            gesture_color = (0, 100, 255)
            distance_text = "Distance: --"

            # If hand landmarks are detected
            if result.multi_hand_landmarks:
                # Use the first detected hand
                hand_landmarks = result.multi_hand_landmarks[0]

                # Draw hand landmarks and connections on the frame
                mp_draw.draw_landmarks(
                    frame,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS
                )

                # Get frame height and width
                h, w, _ = frame.shape

                # ===================== STEP 2: DISTANCE CALCULATION =====================
                # Get thumb tip landmark (ID = 4)
                thumb = hand_landmarks.landmark[4]

                # Get index finger tip landmark (ID = 8)
                index = hand_landmarks.landmark[8]

                # Convert normalized coordinates to pixel coordinates
                x1, y1 = int(thumb.x * w), int(thumb.y * h)
                x2, y2 = int(index.x * w), int(index.y * h)

                # Calculate Euclidean distance between thumb and index finger
                distance = math.hypot(x2 - x1, y2 - y1)

                # Draw circle on thumb tip
                cv2.circle(frame, (x1, y1), 8, (0, 255, 0), -1)

                # Draw circle on index tip
                cv2.circle(frame, (x2, y2), 8, (0, 255, 0), -1)

                # Draw line between thumb and index finger
                cv2.line(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)

                # Gesture class based on distance thresholds
                gesture_text, gesture_color = classify_gesture(distance)
                distance_text = f"Distance: {int(distance)} px"

            # Display overlay values on screen
            cv2.putText(
                frame,
                distance_text,
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 255),
                2,
            )
            cv2.putText(
                frame,
                gesture_text,
                (20, 75),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                gesture_color,
                2,
            )
            cv2.putText(
                frame,
                "Press q to quit",
                (20, 460),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                1,
            )

            # Show the final output window
            cv2.imshow("Milestone 2 - Gesture Recognition", frame)

            # Press 'q' to exit program
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        # Always release resources
        cap.release()
        cv2.destroyAllWindows()
        hands.close()


if __name__ == "__main__":
    main()
