from tqdm import tqdm 

import torch as th

def compute_confusion_matrix(predictions, targets, n_classes = 80):
    # NOTE: we do not include the background in our calculations. 
    # If you want to include that simply let 'n_classes' be the total number of classes including the background. 
    # Otherwise leave it as the number of classes exluding the background.
    preds = predictions.argmax(dim=1) 

    TP = th.zeros(n_classes)
    TN = th.zeros(n_classes)
    FP = th.zeros(n_classes)
    FN = th.zeros(n_classes)

    for c in range(n_classes):
        TP[c] = th.sum((preds == c) & (targets == c))
        TN[c] = th.sum((preds != c) & (targets != c))
        FP[c] = th.sum((preds == c) & (targets != c))
        FN[c] = th.sum((preds != c) & (targets == c))

    conf_matrix = th.stack([TP, FP, FN, TN])

    return conf_matrix

def compute_metrics_from_cm(confusion_matrix):
    TP = confusion_matrix[0]
    FP = confusion_matrix[1]
    FN = confusion_matrix[2]
    TN = confusion_matrix[3]

    support = TP + FN 
    weights = support / support.sum()

    IoU = (TP / (TP + FP + FN + 1e-10)) 
    dice = (2*TP / (2*TP + FP + FN + 1e-10)) 
    precision = (TP / (TP + FP + 1e-10)) 
    sensitivity = (TP / (TP + FN + 1e-10)) 
    specificity = (TN / (TN + FP + 1e-10)) 
    accuracy = ((TP + TN) / (TP + TN + FP + FN + 1e-10)) 

    mean_IoU = (IoU * weights).sum()
    mean_dice = (dice * weights).sum()
    mean_precision = (precision * weights).sum()
    mean_sensitivity = (sensitivity * weights).sum()
    mean_specificity = (specificity * weights).sum()
    mean_accuracy = (accuracy * weights).sum()

    return mean_IoU, mean_dice, mean_precision, mean_sensitivity, mean_specificity, mean_accuracy

def compute_metrics(predictions, targets, n_classes = 80):
    cm = compute_confusion_matrix(predictions=predictions, targets=targets, n_classes=n_classes)
    return compute_metrics_from_cm(cm)

def compute_final_iou(model, dataloader, n_classes=80): 
    cm = th.zeros((4, n_classes))
    # Compute confusion matrix for entire dataset : 
    for batch in tqdm(dataloader):
        images, annotations = batch
        preds = model.forward(images)['out'] 
        targets = th.stack([targets[i]['semantic_mask'] for i in range(len(targets))])
        cm += compute_confusion_matrix(predictions=preds, targets=targets)
    
    TP = cm[0]
    FP = cm[1]
    FN = cm[2]
    TN = cm[3]

    mean_IoU = (TP / (TP + FP + FN + 1e-10)).mean()
    mean_accuracy = ((TP + TN) / (TP + TN + FP + FN + 1e-10)).mean()

    return mean_IoU, mean_accuracy