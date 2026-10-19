"""Live webcam emotion recognition.

    python -m src.webcam_demo                                  # default model (see src/predict.py)
    python -m src.webcam_demo --config configs/rafdb.yaml      # another trained model
    python -m src.webcam_demo --source video.mp4               # a video file instead of the camera

Pipeline per frame:
    camera frame -> Haar cascade finds faces -> crop the largest -> grayscale -> resize to the
    model's input -> divide by 255 -> CNN -> softmax -> label + confidence drawn on the video.
Press q to quit. On macOS the first run asks permission for your terminal app to use the camera.

Only the largest face is classified, and its probabilities are averaged over the last few frames
so the label does not flicker.
"""

import argparse
import time
from collections import deque

import cv2
import numpy as np

from src.predict import DEFAULT_CONFIG, DEFAULT_PROTOCOL, load_model, predict_probs

GREEN, WHITE, DARK = (80, 200, 80), (255, 255, 255), (30, 30, 30)


def find_largest_face(detector, gray, min_size):
    """Haar cascade face detection. Returns (x, y, w, h) of the biggest face, or None."""
    faces = detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(min_size, min_size))
    if len(faces) == 0:
        return None
    return max(faces, key=lambda f: f[2] * f[3])


def draw_prediction(frame, box, names, probs, fps, ms):
    """Draw the face box, 'Label 91%', a small bar per class and the speed readout."""
    x, y, w, h = box
    best = int(np.argmax(probs))
    cv2.rectangle(frame, (x, y), (x + w, y + h), GREEN, 2)
    label = f"{names[best]} {probs[best]:.0%}"
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
    top = max(y - th - 12, 0)
    cv2.rectangle(frame, (x, top), (x + tw + 10, top + th + 10), GREEN, -1)
    cv2.putText(frame, label, (x + 5, top + th + 3), cv2.FONT_HERSHEY_SIMPLEX, 0.8, DARK, 2)

    # One horizontal bar per class in the top-left corner
    for i, (name, p) in enumerate(zip(names, probs)):
        yy = 20 + i * 22
        cv2.rectangle(frame, (95, yy - 11), (95 + int(p * 120), yy + 3), GREEN if i == best else (160, 160, 160), -1)
        cv2.putText(frame, name, (5, yy), cv2.FONT_HERSHEY_SIMPLEX, 0.45, WHITE, 1, cv2.LINE_AA)
    cv2.putText(frame, f"{fps:4.1f} fps | model {ms:.0f} ms", (5, frame.shape[0] - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, WHITE, 1, cv2.LINE_AA)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default=DEFAULT_CONFIG)
    p.add_argument("--protocol", default=DEFAULT_PROTOCOL, choices=["paper", "clean"])
    p.add_argument("--source", default="0", help="camera index (0 = built-in) or a video file")
    p.add_argument("--smooth", type=int, default=5, help="average the last N predictions (1 = no smoothing)")
    p.add_argument("--min-face", type=int, default=80, help="ignore faces smaller than this many pixels")
    p.add_argument("--save-frame", help="process the first frame with a face, save it here and exit (for testing)")
    args = p.parse_args()

    model, cfg = load_model(args.config, args.protocol)
    names, size = cfg["class_names"], cfg["image"]["size"]
    detector = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    assert not detector.empty(), "could not load the Haar cascade file"

    cap = cv2.VideoCapture(int(args.source) if args.source.isdigit() else args.source)
    if not cap.isOpened():
        raise SystemExit("Could not open the video source. On macOS, allow your terminal app under "
                         "System Settings > Privacy & Security > Camera, then run again.")

    recent = deque(maxlen=max(args.smooth, 1))
    last = time.time()
    fps = 0.0
    print(f"model: {cfg['dataset']} / {args.protocol}. Press q in the video window to quit.")
    is_camera = args.source.isdigit()
    failed_reads = dark_frames = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            # A camera that was only just authorised can return empty frames for a moment while it
            # starts, so a live camera gets about 5 seconds of retries. A video file just ended.
            failed_reads += 1
            if is_camera and failed_reads < 50:
                time.sleep(0.1)
                continue
            print("Stopped: the camera gave no frames (check System Settings > Privacy & Security > "
                  "Camera, and that no other app is using it)." if is_camera else "End of video.")
            break
        failed_reads = 0
        if is_camera:
            frame = cv2.flip(frame, 1)  # mirror a live camera so it behaves like a mirror
            # An all-black picture means macOS is not letting this app use the camera (common when the
            # command is run from an editor's built-in terminal), or the lens is covered.
            dark_frames = dark_frames + 1 if frame.mean() < 2 else 0
            if dark_frames == 60:
                print("Camera frames are all black. Run this from the macOS Terminal app and allow camera access "
                      "(System Settings > Privacy & Security > Camera), and check the lens is uncovered.")

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        box = find_largest_face(detector, gray, args.min_face)
        if box is not None:
            x, y, w, h = box
            t0 = time.time()
            recent.append(predict_probs(model, frame[y:y + h, x:x + w], size))
            ms = (time.time() - t0) * 1000
            draw_prediction(frame, box, names, np.mean(recent, axis=0), fps, ms)
            if args.save_frame:
                cv2.imwrite(args.save_frame, frame)
                print(f"saved {args.save_frame}: {names[int(np.argmax(np.mean(recent, axis=0)))]}")
                break
        else:
            recent.clear()  # no face: forget old predictions so a new person starts fresh
            cv2.putText(frame, "No face found", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, WHITE, 2)

        now = time.time()
        fps = 0.9 * fps + 0.1 / max(now - last, 1e-6)  # smoothed frames per second
        last = now
        cv2.imshow("Facial emotion recognition (q to quit)", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
