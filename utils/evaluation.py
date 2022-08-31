import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix


# SR : Segmentation Result
# GT : Ground Truth


def confusion(SR, GT):
    # SR = SR.view(-1)
    # GT = GT.view(-1)
    confusion_vector = SR / GT
    # 示例
    # a = torch.Tensor([1, 0, 0, 1])
    # b = torch.Tensor([0, 1, 0, 1])
    # c = a / b
    # tensor([inf, 0., nan, 1.])
    TP = torch.sum(confusion_vector == 1).item()
    FP = torch.sum(confusion_vector == float('inf')).item()
    TN = torch.sum(torch.isnan(confusion_vector)).item()
    FN = torch.sum(confusion_vector == 0).item()

    return TP, FP, TN, FN


def get_result(SR, GT, threshold=0.5):  # 没用到gpu版本
    SR[SR > threshold] = 1
    SR[SR < 1] = 0
    confusion = confusion_matrix(GT, SR)
    TP = confusion[1, 1]
    TN = confusion[0, 0]
    FP = confusion[0, 1]
    FN = confusion[1, 0]

    acc = (TP + TN) / (float(TP + TN + FP + FN) + 1e-6)
    sensitivity = TP / (float(TP + FN) + 1e-6)
    Specificity = TN / (float(TN + FP) + 1e-6)
    precision = TP / (float(TP + FP) + 1e-6)
    F1 = 2 * sensitivity * precision / (sensitivity + precision + 1e-6)
    JS = sum((SR + GT) == 2) / (sum((SR + GT) >= 1) + 1e-6)
    DC = 2 * sum((SR + GT) == 2) / (sum(SR) + sum(GT) + 1e-6)
    IOU = TP / (float(TP + FP + FN) + 1e-6)

    # print('Accuracy:', acc)
    # print('Sensitivity:', sensitivity)
    # print('Specificity:', Specificity)
    # print('precision:', precision)
    # print('F1', F1)
    # print('JS', JS)
    # print('DC', DC)
    # print('IOU', IOU)

    return acc, sensitivity, Specificity, precision, F1, JS, DC, IOU


def get_result_gpu(SR, GT, threshold=0.5):  # gpu版本
    SR[SR > threshold] = 1
    SR[SR < 1] = 0
    TP, FP, TN, FN = confusion(SR, GT)

    acc = (TP + TN) / (float(TP + TN + FP + FN) + 1e-6)
    sensitivity = TP / (float(TP + FN) + 1e-6)
    specificity = TN / (float(TN + FP) + 1e-6)
    precision = TP / (float(TP + FP) + 1e-6)
    F1 = 2 * sensitivity * precision / (sensitivity + precision + 1e-6)
    JS = TP / (float(FP + TP + FN) + 1e-6)
    # DC = 2*TP / (float(FP+2*TP+FN) + 1e-6)
    DC = 2 * ((SR * GT).sum() + 1) / (SR.sum() + GT.sum() + 1)  # 用这种方法算全身的dice
    DC = DC.item()
    IOU = TP / (float(TP + FP + FN) + 1e-6)

    return acc, sensitivity, specificity, precision, F1, JS, DC, IOU


def get_accuracy(SR, GT, threshold=0.5):
    SR = SR.view(-1)
    GT = GT.view(-1)
    SR = SR > threshold
    GT = GT == torch.max(GT)
    corr = torch.sum(SR == GT)

    # tensor_size = SR.size(0)*SR.size(1)*SR.size(2)*SR.size(3)
    acc = float(corr) / float(SR.size(0))

    return acc


def get_sensitivity(SR, GT, threshold=0.5):
    # Sensitivity == Recall
    SR = SR.view(-1)
    GT = GT.view(-1)
    SR = SR > threshold
    GT = GT == torch.max(GT)

    # TP : True Positive
    # FN : False Negative
    TP = ((SR == 1) + (GT == 1)) == 2
    FN = ((SR == 0) + (GT == 1)) == 2

    SE = float(torch.sum(TP)) / (float(torch.sum(TP + FN)) + 1e-6)

    return SE


def get_specificity(SR, GT, threshold=0.5):
    SR = SR.view(-1)
    GT = GT.view(-1)
    SR = SR > threshold
    GT = GT == torch.max(GT)

    # TN : True Negative
    # FP : False Positive
    TN = ((SR == 0) + (GT == 0)) == 2
    FP = ((SR == 1) + (GT == 0)) == 2

    SP = float(torch.sum(TN)) / (float(torch.sum(TN + FP)) + 1e-6)

    return SP


def get_precision(SR, GT, threshold=0.5):
    SR = SR.view(-1)
    GT = GT.view(-1)
    SR = SR > threshold
    GT = GT == torch.max(GT)

    # TP : True Positive
    # FP : False Positive
    TP = ((SR == 1) + (GT == 1)) == 2
    FP = ((SR == 1) + (GT == 0)) == 2

    PC = float(torch.sum(TP)) / (float(torch.sum(TP + FP)) + 1e-6)

    return PC


def get_F1(SR, GT, threshold=0.5):
    # Sensitivity == Recall
    SR = SR.view(-1)
    GT = GT.view(-1)
    SE = get_sensitivity(SR, GT, threshold=threshold)
    PC = get_precision(SR, GT, threshold=threshold)

    F1 = 2 * SE * PC / (SE + PC + 1e-6)

    return F1


def get_JS(SR, GT, threshold=0.5):
    # JS : Jaccard similarity
    SR = SR.view(-1)
    GT = GT.view(-1)
    SR = SR > threshold
    GT = GT == torch.max(GT)

    Inter = torch.sum((SR + GT) == 2)
    Union = torch.sum((SR + GT) >= 1)

    JS = float(Inter) / (float(Union) + 1e-6)

    return JS


def get_DC(SR, GT, threshold=0.5):
    # DC : Dice Coefficient
    SR = SR.view(-1)
    GT = GT.view(-1)
    SR = SR > threshold
    GT = GT == torch.max(GT)

    Inter = torch.sum((SR + GT) == 2)
    DC = float(2 * Inter) / (float(torch.sum(SR) + torch.sum(GT)) + 1e-6)

    return DC


def get_IOU(SR, GT, threshold=0.5):
    SR = SR.view(-1)
    GT = GT.view(-1)
    SR = SR > threshold
    GT = GT == torch.max(GT)

    # TP : True Positive
    # FP : False Positive
    # FN : False Negative
    TP = ((SR == 1) + (GT == 1)) == 2
    FP = ((SR == 1) + (GT == 0)) == 2
    FN = ((SR == 0) + (GT == 1)) == 2

    IOU = float(torch.sum(TP)) / (float(torch.sum(TP + FP + FN)) + 1e-6)

    return IOU


class SoftDiceLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(SoftDiceLoss, self).__init__()

    def single_dice_cal(self, img, targ):
        # score = 0
        smooth = 1e-5
        img = img.view(img.shape[0], -1)
        targ = targ.view(img.shape[0], -1)  # [B,N]
        intersection = (img * targ).sum(1)
        union = img.sum(1) + targ.sum(1)
        # union[union == 0] = 2 * intersection[union == 0]    # 若union为0 则dice为1
        dice = (2. * intersection + smooth) / (union + smooth)
        score = 1 - dice.sum() / img.shape[0]
        return score

    def forward(self, probs, targets):
        num = targets.size(0)
        # probs = F.sigmoid(logits)
        # m1 = probs.view(num, -1)
        # m2 = targets.view(num, -1)
        m1 = probs
        m2 = targets
        score = 0
        if m1.shape[1] > 1:
            for channel in range(m1.shape[1]):
                img = m1[:, channel, ...]  # [B,H,W]
                targ = m2[:, channel, ...]
                score += self.single_dice_cal(img, targ)
            score = score / m1.shape[1]
        else:
            score = self.single_dice_cal(m1, m2)
        return score
