import os

import numpy as np

from detection import detection_cast, draw_detections, extract_detections
from metrics import iou_score


class Tracker:
    """Generate detections and build tracklets."""

    def __init__(self, return_images=True, lookup_tail_size=80, labels=None):
        self.return_images = return_images  # Return images or detections?
        self.frame_index = 0
        self.labels = labels  # Tracker label list
        self.detection_history = []  # Saved detection list
        self.last_detected = {}
        self.tracklet_count = 0  # Counter to enumerate tracklets

        # We will search tracklet at last lookup_tail_size frames
        self.lookup_tail_size = lookup_tail_size

    def new_label(self):
        """Get new unique label."""
        self.tracklet_count += 1
        return self.tracklet_count - 1

    def init_tracklet(self, frame):
        """Get new unique label for every detection at frame and return it."""
        # Write code here
        # Use extract_detections and new_label
        min_conf = 0.6
        # detections - numpy int array Cx5 [[label_id, xmin, ymin, xmax, ymax]]
        detections = extract_detections(frame, min_confidence=min_conf)
        det_count = detections.shape[0]
        for i in range(det_count):
            detections[i][0] = self.new_label()
        return detections

    @property
    def prev_detections(self):
        """Get detections at last lookup_tail_size frames from detection_history.

        One detection at one id.
        """
        detections = []
        # Write code here
        last = self.detection_history[-self.lookup_tail_size:]
        # detection_label_to_last_frame = {}
        detection_label_to_last_detection = {}
        for i in range(len(last)):
            curr_detections = last[i]
            det_count = curr_detections.shape[0]
            for j in range(det_count):
                detection = curr_detections[j]
                label = detection[0]
                # detection_label_to_last_frame[label] = i
                detection_label_to_last_detection[label] = detection

        for label in detection_label_to_last_detection.keys():
            detections.append(detection_label_to_last_detection[label])
        return detection_cast(detections)

    def bind_tracklet(self, detections):
        """Set id at first detection column.

        Find best fit between detections and previous detections.

        detections: numpy int array Cx5 [[label_id, xmin, ymin, xmax, ymax]]
        return: binded detections numpy int array Cx5 [[tracklet_id, xmin, ymin, xmax, ymax]]
        """
        detections = detections.copy()
        prev_detections = self.prev_detections

        # Write code here

        # Step 1: calc pairwise detection IOU
        threshold = 0.5

        # prev_det_id_to_iou = {}
        new_det_i_to_tracklet_id = {}
        unbinded = set()
        binded_prev = set()
        for new_det_i in range(detections.shape[0]):
            ious = []
            for prev_det_i in range(prev_detections.shape[0]):
                prev_id = prev_detections[prev_det_i][0]
                iou_with_prev_id = (iou_score(detections[new_det_i][1:], prev_detections[prev_det_i][1:]), prev_id)
                ious.append(iou_with_prev_id)

            # Step 2: sort IOU list
            ious.sort(key=lambda x: -x[0])

            # Step 3: fill detections[:, 0] with best match
            # One matching for each id
            bind_found = False
            for iou_with_prev_id in ious:
                prev_id = iou_with_prev_id[1]
                iou = iou_with_prev_id[0]
                if iou < threshold:
                    # unbinded.add(new_det_i)
                    break

                if prev_id not in binded_prev:
                    bind_found = True
                    new_det_i_to_tracklet_id[new_det_i] = prev_id
                    detections[new_det_i][0] = prev_id
                    binded_prev.add(prev_id)

            if not bind_found:
                unbinded.add(new_det_i)

        # Step 4: assign new tracklet id to unmatched detections
        for unbinded_i in unbinded:
            detections[unbinded_i][0] = self.new_label()

        return detection_cast(detections)

    def save_detections(self, detections):
        """Save last detection frame number for each label."""
        for label in detections[:, 0]:
            self.last_detected[label] = self.frame_index

    def update_frame(self, frame):
        if not self.frame_index:
            # First frame should be processed with init_tracklet function
            detections = self.init_tracklet(frame)
        else:
            # Every Nth frame should be processed with CNN (very slow)
            # First, we extract detections
            detections = extract_detections(frame, labels=self.labels, min_confidence=0.4)
            # Then bind them with previous frames
            # Replacing label id to tracker id is performing in bind_tracklet function
            detections = self.bind_tracklet(detections)

        # After call CNN we save frame number for each detection
        self.save_detections(detections)
        # Save detections and frame to the history, increase frame counter
        self.detection_history.append(detections)
        self.frame_index += 1

        # Return image or raw detections
        # Image usefull to visualizing, raw detections to metric
        if self.return_images:
            return draw_detections(frame, detections)
        else:
            return detections


def main():
    from moviepy.editor import VideoFileClip

    dirname = os.path.dirname(__file__)
    input_clip = VideoFileClip(os.path.join(dirname, "data", "test.mp4"))

    tracker = Tracker()
    tracker.return_images = True
    input_clip.fl_image(tracker.update_frame).preview()


if __name__ == "__main__":
    main()
