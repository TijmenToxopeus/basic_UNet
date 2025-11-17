import torch
import torch.nn as nn
import torch.nn.functional as F

# ------------------------------------------------------------
# Loss function factory
# ------------------------------------------------------------
def get_loss_function(name):
    name = name.lower()
    if name == "ce":
        return nn.CrossEntropyLoss()
    elif name == "dice":
        return DiceLoss()
    elif name == "ce_dice":
        return CombinedCELossDice()
    elif name == "focal":
        return FocalLoss()
    elif name == "dice_focal":
        return CombinedDiceFocalLoss()
    else:
        raise ValueError(f"❌ Unknown loss function: {name}")


# ------------------------------------------------------------
# Dice Loss (vectorized, efficient)
# ------------------------------------------------------------
class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-6):
        super().__init__()
        self.smooth = smooth

    def forward(self, preds, targets):
        # preds: (B, C, H, W)
        # targets: (B, H, W)
        preds = F.softmax(preds, dim=1)
        num_classes = preds.shape[1]

        # One-hot encode the target for multi-class dice
        target_onehot = F.one_hot(targets, num_classes).permute(0, 3, 1, 2).float()

        # Dice computation
        dims = (0, 2, 3)
        intersection = torch.sum(preds * target_onehot, dims)
        pred_sum = torch.sum(preds, dims)
        target_sum = torch.sum(target_onehot, dims)

        dice = (2 * intersection + self.smooth) / (pred_sum + target_sum + self.smooth)
        loss = 1 - dice.mean()  # mean over classes

        return loss


# ------------------------------------------------------------
# Focal Loss (for multiclass segmentation)
# ------------------------------------------------------------
class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, alpha=0.25):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, preds, targets):
        # preds: (B, C, H, W)
        # targets: (B, H, W)
        ce_loss = F.cross_entropy(preds, targets, reduction="none")  # per-pixel loss

        pt = torch.exp(-ce_loss)  # pt = probability of correct class
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss

        return focal_loss.mean()


# ------------------------------------------------------------
# CE + Dice (balanced)
# ------------------------------------------------------------
class CombinedCELossDice(nn.Module):
    def __init__(self, ce_weight=0.5, dice_weight=0.5):
        super().__init__()
        self.ce = nn.CrossEntropyLoss()
        self.dice = DiceLoss()
        self.ce_weight = ce_weight
        self.dice_weight = dice_weight

    def forward(self, preds, targets):
        return (
            self.ce_weight * self.ce(preds, targets)
            + self.dice_weight * self.dice(preds, targets)
        )


# ------------------------------------------------------------
# Dice + Focal (RECOMMENDED)
# ------------------------------------------------------------
class CombinedDiceFocalLoss(nn.Module):
    def __init__(self, dice_weight=0.5, focal_weight=0.5, gamma=2.0, alpha=0.25):
        super().__init__()
        self.dice = DiceLoss()
        self.focal = FocalLoss(gamma=gamma, alpha=alpha)
        self.dice_weight = dice_weight
        self.focal_weight = focal_weight

    def forward(self, preds, targets):
        return (
            self.dice_weight * self.dice(preds, targets)
            + self.focal_weight * self.focal(preds, targets)
        )
