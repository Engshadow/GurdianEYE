import time
from pathlib import Path

import cv2

from detection.detector import Detector
from ppe_matcher import PPEMatcher, MatcherConfig, summarize
from safety_monitor import SafetyMonitor
from timing import TimingConfig

# PPE monitoring and violation detection
# =========================================
# CONFIG
# =========================================

PROJECT_DIR = Path(__file__).resolve().parent

VIDEO_SOURCE = PROJECT_DIR / "videos" / "test3.mp4"
MODEL_PATH = PROJECT_DIR / "models" / "best.pt"
VIOLATIONS_DIR = PROJECT_DIR / "violations"

CONFIDENCE_THRESHOLD = 0.2
IOU_THRESHOLD = 0.3
YOLO_IMG_SIZE = 1280

DISPLAY_WIDTH = 1900

PLAYBACK_SLOWDOWN = 2.0

WINDOW_NAME = "PPE Detection - Person 2 Pipeline"


# =========================================
# 1. Create Detector
# =========================================

detector = Detector(
    model_path=str(MODEL_PATH),
    confidence_threshold=0.25,
    iou_threshold=IOU_THRESHOLD,
    img_size=YOLO_IMG_SIZE,
    use_tta=True,
    target_classes=(
        "person", "helmet", "hardhat", "hard_hat", "hat",
        "vest", "safety_vest", "safety_vest_1",
    ),
)


# =========================================
# 2. Create PPE Matcher
# =========================================

# =========================================
# 3. Camera / Video
# =========================================

cap = cv2.VideoCapture(VIDEO_SOURCE)

if not cap.isOpened():

    print(
        "Error: Could not open video/camera source."
    )

    exit()


source_fps = cap.get(cv2.CAP_PROP_FPS)

if not source_fps or source_fps <= 0:

    source_fps = 30


timing = TimingConfig(
    violation_seconds=1.0,
    filter_seconds=0.5,
    stability_seconds=0.6,
    source_fps=source_fps,
    playback_slowdown=PLAYBACK_SLOWDOWN,
)

matcher = PPEMatcher(MatcherConfig(
    min_person_confidence=0.30,
    min_ppe_confidence=0.30,
    temporal_filter_frames=timing.temporal_filter_frames,
    stability_frames=timing.stability_frames,
))

monitor = SafetyMonitor(
    violation_duration=timing.violation_seconds,
    screenshot_dir=str(VIOLATIONS_DIR),
    alarm_cooldown=3.0,
    playback_slowdown=PLAYBACK_SLOWDOWN,
    rearm_seconds=2.0,
)


frame_duration = PLAYBACK_SLOWDOWN / source_fps

is_running = True

prev_time = time.time()


# =========================================
# Resize Function
# =========================================

def resize_frame(frame, target_width):

    h, w = frame.shape[:2]

    if w == target_width:

        return frame

    scale = target_width / w

    new_size = (
        target_width,
        int(h * scale)
    )

    return cv2.resize(
        frame,
        new_size,
        interpolation=cv2.INTER_AREA
    )


# =========================================
# Stop Button
# =========================================

stop_requested = False

STOP_BUTTON = (
    10,
    50,
    130,
    90
)


def mouse_callback(
    event,
    x,
    y,
    flags,
    param
):

    global stop_requested

    if event == cv2.EVENT_LBUTTONDOWN:

        x1, y1, x2, y2 = STOP_BUTTON

        if (
            x1 <= x <= x2
            and
            y1 <= y <= y2
        ):

            stop_requested = True


cv2.namedWindow(WINDOW_NAME)

cv2.setMouseCallback(
    WINDOW_NAME,
    mouse_callback
)


# =========================================
# 4. Frame Processing Loop
# =========================================

while True:

    loop_start = time.time()

    if is_running:

        ret, frame = cap.read()

        if not ret:

            print(
                "Video ended or camera disconnected."
            )

            break

        video_time = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0


        # =====================================
        # Resize Frame
        # =====================================

        frame = resize_frame(
            frame,
            DISPLAY_WIDTH
        )


        print(
            "\nOriginal/Processed frame size:",
            frame.shape
        )


        # =====================================
        # 6. Detection
        # =====================================

        detections, results = detector.detect_frame(
            frame
        )


        # =====================================
        # 7. PPE Matching
        # =====================================

        safety_results = matcher.match(
            detections,
            frame_shape=frame.shape
        )


        # =====================================
        # 8. Create Annotated Frame
        # =====================================

        annotated_frame = frame.copy()


        # =====================================
        # 9. Draw Final PPE Status
        # =====================================

        for item in safety_results:

            x1, y1, x2, y2 = map(
                int,
                item["bbox"]
            )

            status = item["status"]

            person_id = item["person_id"]


            # =================================
            # SAFE
            # =================================

            if status == "SAFE":

                color = (
                    0,
                    255,
                    0
                )

                status_text = "SAFE"


            # =================================
            # NO HELMET
            # =================================

            elif status == "NO_HELMET":

                color = (
                    0,
                    0,
                    255
                )

                status_text = "NO HELMET"


            # =================================
            # NO VEST
            # =================================

            elif status == "NO_VEST":

                color = (
                    0,
                    0,
                    255
                )

                status_text = "NO VEST"

            elif status == "CRITICAL_VIOLATION":

                color = (0, 140, 255)
                status_text = "NO HELMET + NO VEST"

            elif status == "UNKNOWN":

                color = (160, 160, 160)
                status_text = "CHECKING"


            # =================================
            # Ignore CRITICAL
            # =================================

            else:

                continue


            # =================================
            # Person Bounding Box
            # =================================

            cv2.rectangle(
                annotated_frame,
                (x1, y1),
                (x2, y2),
                color,
                3
            )


            # =================================
            # Status Text
            # =================================

            cv2.putText(
                annotated_frame,
                f"Person {person_id}: {status_text}",
                (x1, max(30, y1 - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                color,
                2
            )


        # =====================================
        # 10. Safety Monitor
        # =====================================

        alerts = monitor.process(
            safety_results,
            frame=frame,
            video_time=video_time,
        )


        # =====================================
        # 11. Print Alerts
        # =====================================

        for alert in alerts:

            print(
                f" ALERT: Person "
                f"{alert['person_id']} "
                f"- {alert['type']}"
            )


        # =====================================
        # 12. Print Detections
        # =====================================

        print("=" * 30)

        print(
            "FRAME DETECTIONS"
        )

        print("=" * 30)


        if not detections:

            print("(none)")


        for detection in detections:

            print(
                f"{detection['class']}: "
                f"{detection['confidence']:.3f} "
                f"bbox={detection['bbox']}"
            )


        # =====================================
        # 13. PPE Safety Status
        # =====================================

        print("=" * 30)

        print(
            "PPE SAFETY STATUS"
        )

        print("=" * 30)


        for item in safety_results:

            print(
                f"person_id={item['person_id']} "
                f"status={item['status']} "
                f"missing={item['missing_ppe']}"
            )


        # =====================================
        # 14. Summary
        # =====================================

        print(
            "summary:",
            summarize(safety_results)
        )


        # =====================================
        # 15. FPS
        # =====================================

        current_time = time.time()

        actual_fps = (
            1 /
            (current_time - prev_time)
            if current_time != prev_time
            else 0
        )

        prev_time = current_time


        cv2.putText(
            annotated_frame,
            f"FPS: {actual_fps:.1f}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2
        )


        # =====================================
        # 16. STOP Button
        # =====================================

        x1, y1, x2, y2 = STOP_BUTTON


        cv2.rectangle(
            annotated_frame,
            (x1, y1),
            (x2, y2),
            (0, 0, 255),
            -1
        )


        cv2.putText(
            annotated_frame,
            "STOP",
            (x1 + 15, y1 + 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2
        )


        # =====================================
        # 17. Display
        # =====================================

        cv2.imshow(
            WINDOW_NAME,
            annotated_frame
        )


    # =========================================
    # 18. Timing
    # =========================================

    elapsed = time.time() - loop_start

    remaining = frame_duration - elapsed

    wait_ms = max(
        1,
        int(remaining * 1000)
    )


    key = cv2.waitKey(wait_ms) & 0xFF


    if key == ord("q") or stop_requested:

        break


    elif key == ord("p"):

        is_running = not is_running

        print(
            "Paused"
            if not is_running
            else "Resumed"
        )


# =========================================
# 19. Cleanup
# =========================================

cap.release()

cv2.destroyAllWindows()