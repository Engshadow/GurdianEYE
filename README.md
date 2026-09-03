# GurdianEYE

AI-Powered Workplace Safety Monitoring System

GuardianEYE is an AI-powered workplace safety monitoring system designed to automatically monitor workers, detect Personal Protective Equipment (PPE), identify safety violations, and provide evidence for confirmed violations.

The system goes beyond simple PPE detection by combining object detection, PPE-to-worker matching, worker tracking, safety decision logic, and violation evidence capture into one integrated pipeline.

Project Overview

Workplace safety, especially on construction sites, requires continuous monitoring to ensure that workers follow PPE requirements.

Traditional safety monitoring depends heavily on human supervisors who cannot continuously monitor every worker and camera. GuardianEYE provides an automated solution that analyzes video streams and assists safety teams in identifying PPE violations.

The system processes video frames through the following pipeline:

Video Input → YOLO Detection → PPE Matching → Worker Tracking → Safety Logic → Violation Detection → Evidence

Objectives

GuardianEYE aims to:

Automatically detect workers and PPE.
Identify whether workers are wearing the required PPE.
Associate PPE items with the correct workers.
Track workers across video frames.
Detect missing PPE and safety violations.
Reduce dependence on continuous manual monitoring.
Generate evidence for confirmed violations.
Provide safety statistics for the monitored area.
Key Features
Worker Detection

Detects workers in video frames using a YOLO-based object detection model.

PPE Detection

Detects safety equipment such as:

Helmet
Safety Vest
No Helmet
No Safety Vest
PPE-to-Worker Matching

The system associates detected PPE items with the appropriate worker based on their locations and bounding boxes.

This allows GuardianEYE to determine the PPE status of each individual worker instead of only detecting PPE objects independently.

Worker Tracking

Workers can be tracked across consecutive video frames using their assigned IDs.

This allows the system to maintain an individual safety status over time.

Safety Decision Logic

GuardianEYE applies predefined safety rules to determine the status of each worker.

Helmet	Vest	Status
Yes	Yes	SAFE
No	Yes	NO_HELMET
Yes	No	NO_VEST
No	No	NO_HELMET_AND_VEST
Violation Detection

When a worker is identified as not following the required PPE rules, the system records the violation and identifies the missing PPE.

Violation Evidence

Confirmed violations can be saved with a timestamped snapshot to provide evidence for later review.

Safety Statistics

The system generates a summary of monitored workers, including:

Total workers
Safe workers
Workers with violations
Unknown cases
System Architecture
                    +------------------+
                    |   Video / Camera |
                    +--------+---------+
                             |
                             v
                    +------------------+
                    | Frame Processing  |
                    |     OpenCV        |
                    +--------+---------+
                             |
                             v
                    +------------------+
                    |  YOLO Detection   |
                    |  Workers + PPE    |
                    +--------+---------+
                             |
                             v
                    +------------------+
                    |   PPE Matching    |
                    |    PPE -> Worker  |
                    +--------+---------+
                             |
                             v
                    +------------------+
                    | Worker Tracking  |
                    |    Person IDs     |
                    +--------+---------+
                             |
                             v
                    +------------------+
                    |   Safety Logic    |
                    | Safe / Violation  |
                    +--------+---------+
                             |
                  +----------+----------+
                  |                     |
                  v                     v
          +---------------+     +---------------+
          | Alerts / Stats|     |    Evidence   |
          |               |     |    Snapshot   |
          +---------------+     +---------------+
How GuardianEYE Works
1. Video Input

The system receives video from a camera or an uploaded video.

2. Frame Processing

The video is divided into individual frames using OpenCV.

3. Object Detection

The YOLO model detects workers and PPE objects in each frame.

4. PPE Matching

Detected PPE items are associated with the correct worker.

5. Worker Tracking

Workers are assigned IDs so their status can be followed across frames.

6. Safety Decision

The Safety Logic checks the PPE associated with each worker against predefined safety rules.

7. Violation Detection

The system identifies missing PPE and determines whether a worker has a safety violation.

8. Evidence Capture

For confirmed violations, the system can save a timestamped snapshot for review.

9. Safety Summary

The system generates statistics showing the overall safety status of the monitored workers.

AI Model

GuardianEYE uses a YOLO-based object detection model for PPE and worker detection.

The project includes a YOLO11n model (yolo11n.pt) and custom-trained PPE detection weights.

The model is evaluated using standard object detection metrics:

Precision
Recall
mAP@50
mAP@50–95
