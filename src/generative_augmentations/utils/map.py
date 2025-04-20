import torch as th
import numpy as np

def IoU(mask1, mask2):
    intersection = (mask1 & mask2).sum(dim=(1,2))
    union = (mask1 | mask2).sum(dim=(1,2))
    return intersection / union


def average_precision(targets, detections, iou_threshold=0.5, det_mask_threshold = 0.5, num_classes=80):
    true_positives = np.zeros(num_classes)
    false_positives = np.zeros(num_classes)
    false_negatives = np.zeros(num_classes)
    ap = np.zeros(num_classes)
    no_gt = np.zeros(num_classes)

    for class_id in range(num_classes):
        images = {} # Key : image_number, value : dict(key: target/detection, value: mask)
        scores = [] # List of lists, each entry is score and image number 

        for i, (target, detection) in enumerate(zip(targets, detections)):
            # Get the index for the current class label
            idx_target = th.where(target['labels'] == class_id)[0]
            idx_detection = th.where(detection['labels'] == class_id)[0]

            images[i] = {"target": target['masks'][idx_target], "detection": detection["masks"][idx_detection]}
            img_id = [i]*len(idx_detection)
            det_idx = list(range(len(idx_detection)))
            detection_scores = detection['scores'][idx_detection].detach().cpu().numpy().tolist()

            scores += [list(x) for x in zip(detection_scores, img_id, det_idx)]
            no_gt[class_id] += len(idx_target)

        scores = np.array(scores)
        scores_sort = sorted(scores, key=lambda x: x[0])


        no_det = 0 # number of detections processed so far
        for score in scores_sort:
            img_id = int(score[1])
            det_idx = int(score[2])
            no_det += 1

            target_masks = images[img_id]["target"]
            detection_mask = images[img_id]["detection"][det_idx] > det_mask_threshold

            if len(target_masks) == 0:
                false_positives[class_id] += 1
                continue

            # compute IoU with all target masks
            iou = IoU(target_masks, detection_mask)
            iou_max_idx = iou.argmax()
            if iou[iou_max_idx] > iou_threshold:
                true_positives[class_id] += 1
                # remove the matched target mask
                images[img_id]["target"] = th.cat([images[img_id]["target"][:iou_max_idx], images[img_id]["target"][iou_max_idx+1:]])
                
                # update the average precision for the class
                ap[class_id] += (true_positives[class_id] / no_det) / no_gt[class_id]
            else:
                false_positives[class_id] += 1
            
        false_negatives[class_id] = no_gt[class_id] - true_positives[class_id]

    return (ap * no_gt).sum() / no_gt.sum()







