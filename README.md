#Drone Monitoring & Decision Support System
Inspired by iDEX DISC 14 Challenge 16 — Indian Army
#Problem Statement
The Indian Army currently lacks a centralised system to monitor multiple drone feeds simultaneously. Drones, birds, and aircraft are monitored in isolation with no unified command view, slowing down decision-making in tactical battle areas.
What This System Does

Detects and classifies objects into 3 classes — Drone, Bird, Aircraft — across multiple live video feeds simultaneously
Tracks each object with a persistent ID across frames using ByteTrack, so the same drone is remembered even if it temporarily leaves the frame
Displays all feeds in a real-time command dashboard with a live threat table showing object ID, class, confidence and timestamp
Runs each feed in its own thread so multiple cameras operate independently without blocking each other
Logs every new detection event with coordinates, confidence and time

#Model

#Architecture: YOLOv11 Object Detection
Trained on merged dataset: NIT Rourkela UAV dataset + Subhranil Dey Drone-Bird-Aircraft dataset (~15,000 images)
Augmentation: horizontal flip, brightness variation
Performance: Drone 90% mAP, Aircraft 86% mAP, Bird 74% mAP

#Tech Stack
Python · YOLOv11 · ByteTrack · Supervision · Streamlit · OpenCV · Threading
How to Run
pip install ultralytics supervision streamlit opencv-python
streamlit run app.py
Roadmap / Future Work

Geospatial map interface — operator selects a region, system shows all drone feeds from that area plotted on a live map
Threat scoring engine — behavioral analysis to classify each drone as FRIENDLY / MONITOR / THREAT based on zone proximity, speed and registry
Alert logging — automatic CSV export of all threat events for post-incident analysis
Integration with live RTSP streams from actual UAS hardware via MAVLink

Known Limitations

Currently validated on public drone footage datasets, not live military hardware feeds
Bird classification at 74% mAP — improvement needed for high-altitude low-visibility scenarios
Threat scoring is rule-based in current version, not ML-driven

#Dataset Credits

NIT Rourkela Drone Detection Dataset
Subhranil Dey Drone Detection Dataset
