import cv2
import mediapipe as mp

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils


def main():
    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        model_complexity=0,
        min_detection_confidence=0.6,
        min_tracking_confidence=0.6,
    )

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not access webcam.")
        hands.close()
        raise SystemExit(1)

    cap.set(3, 640)
    cap.set(4, 480)

    failed_reads = 0
    max_failed_reads = 30

    try:
        while True:
            success, frame = cap.read()
            if not success:
                failed_reads += 1
                if failed_reads >= max_failed_reads:
                    print("Warning: Failed to read frame from webcam repeatedly.")
                    break
                continue

            failed_reads = 0
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = hands.process(rgb)

            if result.multi_hand_landmarks:
                for hand_landmarks in result.multi_hand_landmarks:
                    mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            cv2.putText(
                frame,
                "Milestone 1: Hand Detection",
                (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 255),
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
            cv2.imshow("Milestone 1 - Hand Detection", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()
        hands.close()


if __name__ == "__main__":
    main()
