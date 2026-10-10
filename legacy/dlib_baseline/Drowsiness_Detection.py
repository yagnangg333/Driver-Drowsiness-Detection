"""Command-line driver drowsiness detector using an OpenCV window.

Usage:
    python Drowsiness_Detection.py [--threshold 0.25] [--frames 20] [--camera 0] [--no-alarm]

Press 'q' or Esc to quit.
"""

import argparse
import sys

import cv2
import imutils

from detector import (
    DEFAULT_CONSEC_FRAMES,
    DEFAULT_EAR_THRESHOLD,
    FRAME_WIDTH,
    Alarm,
    DrowsinessDetector,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Real-time driver drowsiness detection")
    parser.add_argument(
        "--threshold",
        type=float,
        default=DEFAULT_EAR_THRESHOLD,
        help="eye aspect ratio below which eyes count as closed (default: %(default)s)",
    )
    parser.add_argument(
        "--frames",
        type=int,
        default=DEFAULT_CONSEC_FRAMES,
        help="consecutive closed-eye frames before alerting (default: %(default)s)",
    )
    parser.add_argument("--camera", type=int, default=0, help="camera index (default: 0)")
    parser.add_argument("--no-alarm", action="store_true", help="disable the alert sound")
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    try:
        detector = DrowsinessDetector(args.threshold, args.frames)
    except FileNotFoundError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    alarm = None if args.no_alarm else Alarm()

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        print(f"Error: could not open camera {args.camera}", file=sys.stderr)
        return 1

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("Error: failed to read a frame from the camera", file=sys.stderr)
                break

            frame = imutils.resize(frame, width=FRAME_WIDTH)
            result = detector.process(frame)

            if alarm and result.drowsy:
                alarm.start()
            elif alarm:
                alarm.stop()

            cv2.imshow("Driver Drowsiness Detection (press q to quit)", frame)
            if (cv2.waitKey(1) & 0xFF) in (ord("q"), 27):
                break
    finally:
        if alarm:
            alarm.stop()
        cap.release()
        cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    sys.exit(main())
