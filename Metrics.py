import torch as t
from sklearn.metrics import roc_auc_score
import torch.nn.functional as F



class MAE(object):
    def __init__(self):
        super(MAE, self).__init__()
        self.name = 'MAE'

    def compute(self, answer, label):
        assert len(answer) == len(label)
        answer = t.Tensor(answer).squeeze(-1)
        label = t.Tensor(label)
        MAE = F.l1_loss(answer, label, reduction = 'mean')
        return MAE.item()

class RMSE(object):
    def __init__(self):
        super(RMSE, self).__init__()
        self.name = 'RMSE'

    def compute(self, answer, label):
        assert len(answer) == len(label)
        answer = t.Tensor(answer).squeeze(-1)
        label = t.Tensor(label)

        RMSE = F.mse_loss(answer, label, reduction = 'mean').sqrt()
        SE = F.mse_loss(answer, label, reduction='none')
        MSE = SE.mean()
        return RMSE.item()
