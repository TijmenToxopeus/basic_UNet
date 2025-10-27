import torch
import torch.nn as nn
import torch.nn.functional as F

def get_loss_function(name):
    if name == "ce":
        return nn.CrossEntropyLoss()
    elif name == "dice":
        return DiceLoss()
    elif name == "ce_dice":
        return CombinedCELossDice()
    else:
        raise ValueError(f"Unknown loss {name}")

class DiceLoss(nn.Module):
    def forward(self, preds, targets, smooth=1e-6):
        preds = F.softmax(preds, dim=1)
        num_classes = preds.shape[1]
        loss = 0
        for c in range(num_classes):
            pred_c = preds[:, c, :, :]
            target_c = (targets == c).float()
            intersection = (pred_c * target_c).sum()
            dice = (2. * intersection + smooth) / (pred_c.sum() + target_c.sum() + smooth)
            loss += 1 - dice
        return loss / num_classes

class CombinedCELossDice(nn.Module):
    def __init__(self):
        super().__init__()
        self.ce = nn.CrossEntropyLoss()
        self.dice = DiceLoss()

    def forward(self, preds, targets):
        return 0.5 * self.ce(preds, targets) + 0.5 * self.dice(preds, targets)
