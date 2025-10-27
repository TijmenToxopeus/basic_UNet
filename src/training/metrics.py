import torch
import torch.nn.functional as F

def dice_score(preds, targets, num_classes, per_class=False, epsilon=1e-6):
    """Compute Dice score. If per_class=True, returns list of Dice for each class."""
    preds = torch.argmax(preds, dim=1)  # [B, H, W]
    dice_per_class = []
    for c in range(num_classes):
        pred_c = (preds == c).float()
        target_c = (targets == c).float()
        intersection = (pred_c * target_c).sum()
        union = pred_c.sum() + target_c.sum()
        dice = (2.0 * intersection + epsilon) / (union + epsilon)
        dice_per_class.append(dice.item())

    if per_class:
        return dice_per_class
    return sum(dice_per_class) / num_classes


def iou_score(preds, targets, num_classes, per_class=False, epsilon=1e-6):
    """Compute IoU score. If per_class=True, returns list of IoU for each class."""
    preds = torch.argmax(preds, dim=1)
    iou_per_class = []
    for c in range(num_classes):
        pred_c = (preds == c).float()
        target_c = (targets == c).float()
        intersection = (pred_c * target_c).sum()
        union = pred_c.sum() + target_c.sum() - intersection
        iou = (intersection + epsilon) / (union + epsilon)
        iou_per_class.append(iou.item())

    if per_class:
        return iou_per_class
    return sum(iou_per_class) / num_classes
