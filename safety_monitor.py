import os
import time
import cv2


VIOLATION_STATES = (
    "NO_HELMET",
    "NO_VEST",
    "CRITICAL_VIOLATION",
)

try:
    import winsound
except ImportError:
    winsound = None


class SafetyMonitor:

    def __init__(
        self,
        violation_duration=.5,
        screenshot_dir="violations",
        alarm_cooldown=3.0,
        playback_slowdown=1.0,
        rearm_seconds=2.0,
    ):
        self.violation_duration = violation_duration
        self.screenshot_dir = screenshot_dir
        self.alarm_cooldown = alarm_cooldown
        self.playback_slowdown = max(0.01, playback_slowdown)
        self.rearm_seconds = rearm_seconds

        # Create screenshot folder
        os.makedirs(
            self.screenshot_dir,
            exist_ok=True
        )

        # Track violations
        self.violation_start = {}
        self.last_alarm_time = float("-inf")
        self.safe_since = {}
        self.violation_history = []

        # Store persons who already received an alert
        self.alerted_persons = set()

    # ============================================================
    # PROCESS SAFETY RESULTS
    # ============================================================

    def process(
        self,
        safety_results,
        frame=None,
        restricted_zone=None,
        video_time=None,
    ):
        current_time = (
            video_time
            if video_time is not None
            else time.time() / self.playback_slowdown
        )
        alerts = []

        current_person_ids = set()

        for person in safety_results:

            person_id = person["person_id"]
            status = person["status"]
            bbox = person["bbox"]

            current_person_ids.add(person_id)

            # ====================================================
            # SAFETY VIOLATION
            # ====================================================

            if status in VIOLATION_STATES:
                self.safe_since.pop(person_id, None)

                # Start violation timer
                if person_id not in self.violation_start:
                    self.violation_start[person_id] = current_time

                duration = (
                    current_time
                    - self.violation_start[person_id]
                )

                # Wait until violation lasts 1.5 seconds
                if duration >= self.violation_duration:

                    if person_id not in self.alerted_persons:

                        alert = {
                            "person_id": person_id,
                            "type": status,
                            "duration": duration,
                            "bbox": bbox,
                            "timestamp": current_time
                        }

                        # ====================================================
                        # ALARM
                        # ====================================================

                        if current_time - self.last_alarm_time >= self.alarm_cooldown:
                            self.trigger_alarm()
                            self.last_alarm_time = current_time

                        # ====================================================
                        # SCREENSHOT
                        # ====================================================

                        if frame is not None:

                            screenshot_path = self.save_screenshot(
                                frame,
                                person_id,
                                status,
                                bbox
                            )

                            alert["screenshot"] = screenshot_path

                        # Store alert
                        alerts.append(alert)

                        self.violation_history.append(
                            alert
                        )

                        # Mark person as already alerted
                        self.alerted_persons.add(
                            person_id
                        )

            else:

                # Person became safe
                self.violation_start.pop(
                    person_id,
                    None
                )
                if person_id in self.alerted_persons:
                    safe_since = self.safe_since.setdefault(person_id, current_time)
                    if current_time - safe_since >= self.rearm_seconds:
                        self.alerted_persons.discard(person_id)
                        self.safe_since.pop(person_id, None)

           

        # ========================================================
        # REMOVE DISAPPEARED PERSONS
        # ========================================================

        old_ids = list(
            self.violation_start.keys()
        )

        for person_id in old_ids:

            if person_id not in current_person_ids:

                self.violation_start.pop(
                    person_id,
                    None
                )

        for person_id in list(self.safe_since):
            if person_id not in current_person_ids:
                self.safe_since.pop(person_id, None)

        return alerts

    # ============================================================
    # DRAW SAFETY BOUNDING BOX
    # ============================================================

    def draw_safety_box(
        self,
        frame,
        safety_results
    ):

        for person in safety_results:

            bbox = person["bbox"]
            status = person["status"]
            person_id = person["person_id"]

            x1, y1, x2, y2 = map(
                int,
                bbox
            )

            # ====================================================
            # RED = VIOLATION
            # GREEN = SAFE
            # ====================================================

            if status in (
                "NO_HELMET",
                "NO_VEST",
                "CRITICAL_VIOLATION",
                "RESTRICTED_ZONE"
            ):

                color = (0, 0, 255)

            else:

                color = (0, 255, 0)

            # ====================================================
            # DRAW BOUNDING BOX
            # ====================================================

            cv2.rectangle(
                frame,
                (x1, y1),
                (x2, y2),
                color,
                3
            )

            # ====================================================
            # LABEL
            # ====================================================

            label = f"Person {person_id}: {status}"

            cv2.putText(
                frame,
                label,
                (x1, max(25, y1 - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                color,
                2
            )

        return frame

    # ============================================================
    # ALARM
    # ============================================================

    def trigger_alarm(self):

        print(" SAFETY ALARM!")

        if winsound is not None:

            try:

                winsound.Beep(
                    1000,
                    500
                )

            except Exception:

                pass

    # ============================================================
    # SAVE VIOLATION SCREENSHOT
    # ============================================================

    def save_screenshot(
        self,
        frame,
        person_id,
        violation_type,
        bbox
    ):

        # ========================================================
        # GET ORIGINAL BOUNDING BOX
        # ========================================================

        x1, y1, x2, y2 = map(
            int,
            bbox
        )

        # ========================================================
        # FRAME DIMENSIONS
        # ========================================================

        h, w = frame.shape[:2]

        # ========================================================
        # CALCULATE PERSON CENTER
        # ========================================================

        center_x = (
            x1 + x2
        ) // 2

        center_y = (
            y1 + y2
        ) // 2

        # ========================================================
        # MAKE SCREENSHOT LARGER
        # ========================================================

        scale = 2.0

        person_width = x2 - x1
        person_height = y2 - y1

        new_width = int(
            person_width * scale
        )

        new_height = int(
            person_height * scale
        )

        # ========================================================
        # EXPAND BOUNDING BOX
        # ========================================================

        x1 = max(
            0,
            center_x - new_width // 2
        )

        y1 = max(
            0,
            center_y - new_height // 2
        )

        x2 = min(
            w,
            center_x + new_width // 2
        )

        y2 = min(
            h,
            center_y + new_height // 2
        )

        # ========================================================
        # CROP PERSON
        # ========================================================

        person_crop = frame[
            y1:y2,
            x1:x2
        ]

        # ========================================================
        # CHECK CROP
        # ========================================================

        if person_crop.size == 0:

            print(
                "⚠️ Could not crop violating person."
            )

            return None

        # ========================================================
        # FILE NAME
        # ========================================================

        timestamp = time.strftime(
            "%Y%m%d_%H%M%S"
        )

        filename = (
            f"violation_"
            f"person{person_id}_"
            f"{violation_type}_"
            f"{timestamp}.jpg"
        )

        # ========================================================
        # FULL PATH
        # ========================================================

        path = os.path.join(
            self.screenshot_dir,
            filename
        )

        # ========================================================
        # SAVE SCREENSHOT
        # ========================================================

        cv2.imwrite(
            path,
            person_crop
        )

        print(
            f" Violating person saved: {path}"
        )

        return path

    # ============================================================
    # CHECK RESTRICTED ZONE
    # ============================================================

    def is_inside_restricted_zone(
        self,
        bbox,
        polygon
    ):

        x1, y1, x2, y2 = bbox

        # Person center
        center_x = int(
            (x1 + x2) / 2
        )

        center_y = int(
            (y1 + y2) / 2
        )

        # Check if center is inside polygon
        result = cv2.pointPolygonTest(
            polygon,
            (center_x, center_y),
            False
        )

        return result >= 0

    # ============================================================
    # DRAW RESTRICTED ZONE
    # ============================================================

    def draw_restricted_zone(
        self,
        frame,
        polygon
    ):

        cv2.polylines(
            frame,
            [polygon],
            True,
            (0, 0, 255),
            3
        )

        cv2.putText(
            frame,
            "RESTRICTED ZONE",
            tuple(polygon[0]),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 255),
            2
        )

        return frame

    # ============================================================
    # GET ALERT HISTORY
    # ============================================================

    def get_history(self):

        return self.violation_history

    # ============================================================
    # CLEAR HISTORY
    # ============================================================

    def clear_history(self):

        self.violation_history.clear()

        self.violation_start.clear()

        self.last_alarm_time = float("-inf")
        self.safe_since.clear()

        self.alerted_persons.clear()
