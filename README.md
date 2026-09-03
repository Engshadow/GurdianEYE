# GurdianEYE


PPE Detection & Smart Safety Monitoring System
Project Overview

PPE Detection & Smart Safety Monitoring System is an AI-powered workplace safety monitoring system that uses YOLO and Computer Vision to detect workers and their Personal Protective Equipment (PPE), determine each worker's safety status, and automatically generate alerts and visual evidence for safety violations.

The system is designed to help improve workplace safety by automatically monitoring PPE compliance and reducing the need for continuous manual inspection.

Main Objective

The main objective of this project is to develop a real-time safety monitoring system that can detect workers, identify their PPE, determine safety violations, and provide immediate alerts with visual evidence.

The system follows the process:

Detect
   ↓
Match
   ↓
Classify
   ↓
Alert
   ↓
Record
System Architecture
                    Camera / Video
                          |
                       OpenCV
                          |
                      YOLO Model
                          |
             +------------+------------+
             |            |            |
          Person       Helmet         Vest
             |            |            |
             +------------+------------+
                          |
                    PPE Matching
                          |
                  Safety Classification
                          |
             +------------+------------+
             |                         |
           SAFE                   VIOLATION
                                       |
                         +-------------+-------------+
                         |             |             |
                       Alarm      Screenshot        Log
                         |             |             |
                         +-------------+-------------+
                                       |
                              Desktop Application
Key Features
AI-Based PPE Detection

The system uses a YOLO-based object detection model to detect workers and PPE equipment.

Depending on the trained dataset, the supported classes may include:

Person
Helmet
Vest
No Helmet
No Vest
Person-PPE Matching

The system determines which helmet and vest belong to each detected worker using bounding-box relationships and spatial information.

Example:

Person #1
   |
   +-- Helmet
   |
   +-- Vest

Status: SAFE
Safety Violation Detection

The system can detect different types of safety violations, including:

NO HELMET
NO VEST
NO HELMET + NO VEST
RESTRICTED ZONE VIOLATION
Violation Persistence

To reduce false alarms, the system does not necessarily trigger an alarm when a violation appears in only one frame.

The violation can be required to remain present for a specific period before an alert is generated.

Violation Detected
        |
        v
Persists for Required Duration?
        |
       YES
        |
        v
      ALERT
Violation Evidence

When a confirmed violation occurs, the system captures a screenshot of the violation and stores it for later review.

Example:

violations/
    violation_001.jpg
    violation_002.jpg
    violation_003.jpg
Safety Alarm

The system can trigger an audio alarm when a confirmed safety violation is detected.
