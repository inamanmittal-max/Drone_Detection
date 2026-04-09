import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import time
import cv2
import supervision as sv
from ultralytics import YOLO
import numpy as np
import threading
from collections import deque


class DroneProcessor:
    """Processes video frames for object detection and tracking."""

    def __init__(self, source_path="videoplayback.mp4", max_log_size=1000):
        """
        Initialize the processor with YOLO model and tracker.
        
        Args:
            source_path: Path to the video source for FPS extraction
            max_log_size: Maximum number of log entries to keep (FIFO buffer)
        """
        self.model = YOLO("best.pt")

        video_info = sv.VideoInfo.from_video_path(source_path)
        self.fps = video_info.fps

        self.tracker = sv.ByteTrack(
            lost_track_buffer=60,
            minimum_matching_threshold=0.8,
            minimum_consecutive_frames=3
        )

        self.seen_ids = set()
        self.log_data = deque(maxlen=max_log_size)  # Limited memory with FIFO

        self.last_detections = None  # for hybrid tracking

        self.box_annotator = sv.BoxAnnotator()
        self.label_annotator = sv.LabelAnnotator()

    def _log_entry(self, tracker_id, class_name, confidence, frame_idx, video_time, 
                  center_x, center_y, width, height, event):
        """Helper method to reduce code duplication in logging."""
        self.log_data.append({
            "event": event,
            "id": tracker_id,
            "class": class_name,
            "confidence": confidence,
            "frame": int(frame_idx),
            "video_time_sec": float(video_time),
            "center_x": float(center_x),
            "center_y": float(center_y),
            "width": float(width),
            "height": float(height)
        })

    def process_frame(self, frame: np.ndarray, index: int) -> np.ndarray:
        """
        Process a single frame for object detection and tracking.
        
        Args:
            frame: Input video frame
            index: Frame index
            
        Returns:
            Annotated frame with bounding boxes and labels
        """
        # HYBRID DETECTION - run detection every 2 frames for performance
        if index % 2 == 0:
            results = self.model(frame, verbose=False)[0]
            detections = sv.Detections.from_ultralytics(results)
            detections = detections[detections.confidence > 0.5]
            self.last_detections = detections
        else:
            # Reuse previous detections
            if self.last_detections is None:
                return frame
            detections = self.last_detections

        # Always run tracker
        detections = self.tracker.update_with_detections(detections)

        labels = []
        video_time = index / self.fps
        h, w = frame.shape[:2]

        if detections.tracker_id is not None:
            for i in range(len(detections.tracker_id)):
                class_id = int(detections.class_id[i])
                tracker_id = int(detections.tracker_id[i])
                confidence = float(detections.confidence[i])
                class_name = self.model.names[class_id]

                # Normalize coordinates
                x1, y1, x2, y2 = detections.xyxy[i]
                width = (x2 - x1) / w
                height = (y2 - y1) / h
                center_x = ((x1 + x2) / 2) / w
                center_y = ((y1 + y2) / 2) / h

                # Log new objects
                if tracker_id not in self.seen_ids:
                    self.seen_ids.add(tracker_id)
                    self._log_entry(tracker_id, class_name, confidence, index, video_time,
                                  center_x, center_y, width, height, "NEW_OBJECT")

                # Log periodic updates every 10 frames
                if index % 10 == 0:
                    self._log_entry(tracker_id, class_name, confidence, index, video_time,
                                  center_x, center_y, width, height, "TRACK_UPDATE")

                # Create label
                labels.append(f"ID:{tracker_id} {class_name} ({confidence:.2f})")

        annotated_frame = self.box_annotator.annotate(
            scene=frame.copy(),
            detections=detections
        )

        annotated_frame = self.label_annotator.annotate(
            scene=annotated_frame,
            detections=detections,
            labels=labels
        )

        return annotated_frame


class FeedManager:
    """Manages multiple video feeds with concurrent processing."""

    def __init__(self, sources):
        """
        Initialize feed manager with video sources.
        
        Args:
            sources: Dict mapping feed names to video file paths
        """
        self.sources = sources
        self.processors = {}
        self.latest_frames = {}
        self.lock = threading.Lock()

        # Create processor per feed
        for name, path in sources.items():
            self.processors[name] = DroneProcessor(path)
            self.latest_frames[name] = None

    def start(self):
        """Start processing threads for all feeds."""
        for name, path in self.sources.items():
            thread = threading.Thread(
                target=self.run_feed,
                args=(name, path),
                daemon=True
            )
            thread.start()

    def run_feed(self, name, path):
        """Process a single video feed."""
        cap = cv2.VideoCapture(path)

        if not cap.isOpened():
            print(f"[ERROR] Cannot open {name}")
            return

        print(f"[STARTED] {name}")

        index = 0
        while True:
            ret, frame = cap.read()

            if not ret:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # restart video
                continue

            processor = self.processors[name]
            annotated_frame = processor.process_frame(frame, index)

            # Thread-safe write
            with self.lock:
                self.latest_frames[name] = annotated_frame

            # Print detection count (for debugging)
            if index % 30 == 0:
                num_objects = 0
                if processor.last_detections is not None:
                    num_objects = len(processor.last_detections)
                print(f"{name} | frame {index} | objects: {num_objects}")

            index += 1

        cap.release()


if __name__ == "__main__":
    # Process single video
    # processor = DroneProcessor()
    # sv.process_video(
    #     source_path="videoplayback.mp4",
    #     target_path="output.mp4",
    #     callback=processor.process_frame
    # )

    # Alternatively run multiple feeds using videoplayback.mp4, video2.mp4, and video3.mp4
    sources = {
        "feed_1": "videoplayback.mp4",
        "feed_2": "video2.mp4",
        "feed_3": "video3.mp4"
    }
    manager = FeedManager(sources)
    manager.start()
    while True:
        time.sleep(1)