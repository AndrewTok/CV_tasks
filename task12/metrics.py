def iou_score(bbox1, bbox2):
    """Jaccard index or Intersection over Union.

    https://en.wikipedia.org/wiki/Jaccard_index

    bbox: [xmin, ymin, xmax, ymax]
    """

    assert len(bbox1) == 4
    assert len(bbox2) == 4
    first_start_row = bbox1[0]
    first_start_col = bbox1[1]
    first_end_row = bbox1[2]#+first_bbox[2]
    first_end_col = bbox1[3] #first_bbox[1]+first_bbox[3]

    second_start_row = bbox2[0]
    second_start_col = bbox2[1]
    second_end_row = bbox2[2] #second_bbox[0]+second_bbox[2]
    second_end_col = bbox2[3]#s#econd_bbox[1]+second_bbox[3]

    first_rl = first_end_row - first_start_row
    first_cl = first_end_col - first_start_col

    second_rl = second_end_row - second_start_row
    secpnd_cl = second_end_col - second_start_col

    overlap_h = max(0, min(first_end_row, second_end_row) - max(first_start_row, second_start_row))
    overlap_w = max(0, min(first_end_col, second_end_col) - max(first_start_col, second_start_col))

    ov_area = overlap_w*overlap_h

    un_area = first_rl*first_cl + second_rl*secpnd_cl - ov_area

    return ov_area/un_area


def convert(detections):
    import numpy as np

    detections = np.array(detections)
    dct = {}
    for i in range(detections.shape[0]):
        dct[detections[i][0]] = detections[i][1:]
    return dct

def delete_obj(to_remove_ids, dct1, dct2 = None):
    for id in to_remove_ids:
        del dct1[id]
        if dct2 is not None:
            del dct2[id]

def motp(obj, hyp, threshold=0.5):
    """Calculate MOTP

    obj: list
        Ground truth frame detections.
        detections: numpy int array Cx5 [[id, xmin, ymin, xmax, ymax]]

    hyp: list
        Hypothetical frame detections.
        detections: numpy int array Cx5 [[id, xmin, ymin, xmax, ymax]]

    threshold: IOU threshold
    """
    dist_sum = 0  # a sum of IOU distances between matched objects and hypotheses
    match_count = 0

    matches = {}  # matches between object IDs and hypothesis IDs

    # For every frame
    for frame_obj, frame_hyp in zip(obj, hyp):
        # Write code here

        # Step 1: Convert frame detections to dict with IDs as keys

        obj_detections_dict = convert(frame_obj)
        hyp_detections_dict = convert(frame_hyp)

        # Step 2: Iterate over all previous matches
        # If object is still visible, hypothesis still exists
        # and IOU distance > threshold - we've got a match
        # Update the sum of IoU distances and match count
        # Delete matched detections from frame detections

        for obj_id in matches.keys():
            hyp_id = matches[obj_id]
            if hyp_id in hyp_detections_dict.keys() and obj_id in obj_detections_dict.keys():
                iou = iou_score(obj_detections_dict[obj_id], hyp_detections_dict[hyp_id])
                if iou > threshold:
                    dist_sum += iou
                    match_count += 1
                    del obj_detections_dict[obj_id]
                    del hyp_detections_dict[hyp_id]


        # Step 3: Calculate pairwise detection IOU between remaining frame detections
        # Save IDs with IOU > threshold
        ious_with_obj_id_hyp_id = []
        for hyp_id in hyp_detections_dict.keys():

            for obj_id in obj_detections_dict.keys():
                iou = iou_score(obj_detections_dict[obj_id], hyp_detections_dict[hyp_id])
                if iou > threshold:
                    ious_with_obj_id_hyp_id.append((iou, obj_id, hyp_id))
            ious_with_obj_id_hyp_id.sort(key=lambda x: -x[0])

        # Step 4: Iterate over sorted pairwise IOU
        # Update the sum of IoU distances and match count
        # Delete matched detections from frame detections

        binded_obj_id = set()
        binded_hyp_id = set()

        for pairwise_iou in ious_with_obj_id_hyp_id:
            iou = pairwise_iou[0]
            obj_id = pairwise_iou[1]
            hyp_id = pairwise_iou[2]
            if obj_id not in binded_obj_id and hyp_id not in binded_hyp_id:
                binded_obj_id.add(obj_id)
                binded_hyp_id.add(hyp_id)
                dist_sum += iou
                match_count += 1
                # Step 5: Update matches with current matched IDs
                matches[obj_id] = hyp_id

        pass

    # Step 6: Calculate MOTP
    MOTP = dist_sum / match_count

    return MOTP


def motp_mota(obj, hyp, threshold=0.5):
    """Calculate MOTP/MOTA

    obj: list
        Ground truth frame detections.
        detections: numpy int array Cx5 [[id, xmin, ymin, xmax, ymax]]

    hyp: list
        Hypothetical frame detections.
        detections: numpy int array Cx5 [[id, xmin, ymin, xmax, ymax]]

    threshold: IOU threshold
    """

    dist_sum = 0  # a sum of IOU distances between matched objects and hypotheses
    match_count = 0
    missed_count = 0
    false_positive = 0
    mismatch_error = 0

    matches = {}  # matches between object IDs and hypothesis IDs

    gt = 0
    # For every frame
    for frame_obj, frame_hyp in zip(obj, hyp):
        # Step 1: Convert frame detections to dict with IDs as keys

        gt += len(frame_obj)

        obj_detections_dict = convert(frame_obj)
        hyp_detections_dict = convert(frame_hyp)

        # Step 2: Iterate over all previous matches
        # If object is still visible, hypothesis still exists
        # and IOU distance > threshold - we've got a match
        # Update the sum of IoU distances and match count
        # Delete matched detections from frame detections


        for obj_id in matches.keys():
            hyp_id = matches[obj_id]
            if hyp_id in hyp_detections_dict.keys() and obj_id in obj_detections_dict.keys():
                iou = iou_score(obj_detections_dict[obj_id], hyp_detections_dict[hyp_id])
                if iou > threshold:
                    dist_sum += iou
                    match_count += 1
                    del obj_detections_dict[obj_id]
                    del hyp_detections_dict[hyp_id]


        # Step 3: Calculate pairwise detection IOU between remaining frame detections
        # Save IDs with IOU > threshold

        ious_with_obj_id_hyp_id = []
        for hyp_id in hyp_detections_dict.keys():
            for obj_id in obj_detections_dict.keys():
                iou = iou_score(obj_detections_dict[obj_id], hyp_detections_dict[hyp_id])
                if iou > threshold:
                    ious_with_obj_id_hyp_id.append((iou, obj_id, hyp_id))
        ious_with_obj_id_hyp_id.sort(key=lambda x: -x[0])

        # Step 4: Iterate over sorted pairwise IOU
        # Update the sum of IoU distances and match count
        # Delete matched detections from frame detections
        binded_obj_id = set()
        binded_hyp_id = set()

        curr_frame_matches = {}
        for pairwise_iou in ious_with_obj_id_hyp_id:
            iou = pairwise_iou[0]
            obj_id = pairwise_iou[1]
            hyp_id = pairwise_iou[2]
            if obj_id not in binded_obj_id and hyp_id not in binded_hyp_id:
                binded_obj_id.add(obj_id)
                binded_hyp_id.add(hyp_id)
                dist_sum += iou
                match_count += 1
                curr_frame_matches[obj_id] = hyp_id
        delete_obj(binded_obj_id, obj_detections_dict)
        delete_obj(binded_hyp_id, hyp_detections_dict)

        # Step 5: If matched IDs contradict previous matched IDs - increase mismatch error
        for obj_id in curr_frame_matches.keys():
            if obj_id not in matches.keys():
                continue
            elif matches[obj_id] != curr_frame_matches[obj_id]:
                mismatch_error += 1

        # Step 6: Update matches with current matched IDs
        for obj_id in curr_frame_matches.keys():
            matches[obj_id] = curr_frame_matches[obj_id]


        # Step 7: Errors
        # All remaining hypotheses are considered false positives
        false_positive += len(hyp_detections_dict)
        # All remaining objects are considered misses
        missed_count += len(obj_detections_dict)
        print("\n","MISSES: " + str(missed_count), "MME: " + str(mismatch_error), "FP: " + str(false_positive))
        pass

    # Step 8: Calculate MOTP and MOTA
    MOTP = dist_sum / match_count
    MOTA = 1 - (missed_count + mismatch_error + false_positive)/gt

    return MOTP, MOTA
