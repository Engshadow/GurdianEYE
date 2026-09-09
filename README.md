# GurdianEYE

## PPE Detection and Smart Safety Monitoring System

GurdianEYE is an artificial intelligence workplace safety system that uses YOLO and computer vision to detect workers and their personal protective equipment.

The system identifies helmets and safety vests, matches PPE to individual workers, classifies safety status, triggers alerts, and saves evidence of confirmed violations.

## Project Title

GurdianEYE: AI-Based PPE Detection and Safety Monitoring System

## Goal

To automatically detect workers and verify helmet and safety-vest compliance in order to identify workplace safety violations quickly.

## Role

Developer responsible for YOLO model integration, computer-vision processing, PPE matching, safety-status classification, temporal filtering, alarm handling, and violation evidence capture.

## Method

The system processes video or camera input using OpenCV. A YOLO model detects people, helmets, and safety vests.

The PPE matcher assigns detected equipment to each worker using bounding-box containment, object position, confidence thresholds, body-region analysis, temporal stability, and worker tracking.

Each worker is classified as one of the following:

- SAFE
- NO_HELMET
- NO_VEST
- CRITICAL_VIOLATION
- UNKNOWN

A violation must remain present for a configured duration before the system triggers an alarm and saves a screenshot.

## Result

The system provides real-time annotated video, worker-level PPE status, detection summaries, safety alarms, and saved screenshots of confirmed violations.

Temporal filtering helps reduce false alerts caused by unstable detections in individual video frames.

The system can identify workers who are fully compliant, missing a helmet, missing a safety vest, or missing both.

## Source Label

Training / Practice

The detector is trained with PPE images and tested using a practice video or camera stream. The system is a prototype and should be validated with site-specific data before being used in a real workplace.

## Evidence Attached

An annotated screenshot showing a detected worker, the worker's bounding box, PPE status, detected helmet or safety vest, and saved violation evidence from the `violations/` folder.

## Caption

Figure 1. GurdianEYE detects a worker and evaluates helmet and safety-vest compliance. The displayed status identifies whether the worker is safe or missing required PPE, while confirmed violations are saved as visual evidence.
