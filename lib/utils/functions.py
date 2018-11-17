from torch.autograd import Function
import torch.nn as nn
import torch
import lib

#https://github.com/fungtion/DSN
#Orthogonality constraint - difference loss
#This looks like normalized and does not produce same result as squared Frobenius norm
class DiffLossNormalized(nn.Module):

    def __init__(self):
        super(DiffLossNormalized, self).__init__()

    def forward(self, input1, input2):
        batch_size = input1.size(0)
        input1 = input1.view(batch_size, -1)
        input2 = input2.view(batch_size, -1)
        input1_l2_norm = torch.norm(input1, p=2).detach()
        input1_l2 = input1.div(input1_l2_norm.expand_as(input1) + 1e-6)
        input2_l2_norm = torch.norm(input2, p=2).detach()
        input2_l2 = input2.div(input2_l2_norm.expand_as(input2) + 1e-6)
        diff_loss = torch.mean((input1_l2.mm(input2_l2.t()).pow(2)))
        return diff_loss


#Orthogonality constraint - difference loss
#This is the correct squared Frobenius norm
class DiffLoss(nn.Module):

    def __init__(self):
        super(DiffLoss, self).__init__()

    def forward(self, input1, input2):
        batch_size = input1.size(0)
        input1 = input1.view(batch_size, -1)
        input2 = input2.view(batch_size, -1)
        mul = input1.mm(input2.t())
        element_wise_norm = torch.norm(mul, p=2, dim=1).detach()
        power = element_wise_norm.pow(2)
        sum = power.sum()
        return sum



class SmoothF2Loss(nn.Module):
    def __init__(self):
        super(SmoothF2Loss, self).__init__()

    def forward(self, input, target):
        return 1 - self.torch_f2_score(target, input)

    def torch_f2_score(self, y_true, y_pred):
        return self.torch_fbeta_score(y_true, y_pred, 2)

    def torch_fbeta_score(self, y_true, y_pred, beta, eps=1e-9):
        beta2 = beta**2

        y_true = y_true.float()
        y_pred = y_pred.float()
        true_positive = (y_pred * y_true).sum(dim=0)
        precision = true_positive.div(y_pred.sum(dim=0).add(eps))
        recall = true_positive.div(y_true.sum(dim=0).add(eps))

        return torch.mean(
            (precision*recall).
                div(precision.mul(beta2) + recall + eps).
                mul(1 + beta2))



class SmoothFpoint5Loss(nn.Module):
    def __init__(self):
        super(SmoothFpoint5Loss, self).__init__()

    def forward(self, input, target):
        return 1 - self.torch_f05_score(target, input)

    def torch_f05_score(self, y_true, y_pred):
        return self.torch_fbeta_score(y_true, y_pred, 0.5)

    def torch_fbeta_score(self, y_true, y_pred, beta, eps=1e-9):
        beta2 = beta**2

        y_true = y_true.float()
        y_pred = y_pred.float()
        true_positive = (y_pred * y_true).sum(dim=0)
        precision = true_positive.div(y_pred.sum(dim=0).add(eps))
        recall = true_positive.div(y_true.sum(dim=0).add(eps))

        return torch.mean(
            (precision*recall).
                div(precision.mul(beta2) + recall + eps).
                mul(1 + beta2))
