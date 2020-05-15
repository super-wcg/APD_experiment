import torch.nn as nn
import torch
import torch.nn.functional as F


class cls_pos(nn.Module):
    def __init__(self):
        super(cls_pos, self).__init__()
        self.bce = nn.BCELoss(reduction='none')

    def forward(self, pos_pred, pos_label):  # 0-gauss 1-mask 2-center
        log_loss = self.bce(pos_pred[:, 0, :, :], pos_label[:, 2, :, :])

        positives = pos_label[:, 2, :, :]
        negatives = pos_label[:, 1, :, :] - pos_label[:, 2, :, :]

        fore_weight = positives * (1.0-pos_pred[:, 0, :, :]) ** 2
        back_weight = negatives * ((1.0-pos_label[:, 0, :, :])**4.0) * (pos_pred[:, 0, :, :]**2.0)

        focal_weight = fore_weight + back_weight
        assigned_box = torch.sum(pos_label[:, 2, :, :])

        cls_loss = 0.01 * torch.sum(focal_weight*log_loss) / max(1.0, assigned_box)

        return cls_loss


class reg_pos(nn.Module):
    def __init__(self):
        super(reg_pos, self).__init__()
        self.smoothl1 = nn.SmoothL1Loss(reduction='none')

    def forward(self, h_pred, h_label):
        l1_loss = h_label[:, 1, :, :]*self.smoothl1(h_pred[:, 0, :, :]/(h_label[:, 0, :, :]+1e-10),
                                                    h_label[:, 0, :, :]/(h_label[:, 0, :, :]+1e-10))
        reg_loss = torch.sum(l1_loss) / max(1.0, torch.sum(h_label[:, 1, :, :]))
        return reg_loss


class offset_pos(nn.Module):
    def __init__(self):
        super(offset_pos, self).__init__()
        self.smoothl1 = nn.SmoothL1Loss(reduction='none')

    def forward(self, offset_pred, offset_label):
        l1_loss = offset_label[:, 2, :, :].unsqueeze(dim=1)*self.smoothl1(offset_pred, offset_label[:, :2, :, :])
        off_loss = 0.1 * torch.sum(l1_loss) / max(1.0, torch.sum(offset_label[:, 2, :, :]))
        return off_loss


class aed_pos(nn.Module):
    def __init__(self):
        super(aed_pos, self).__init__()
        self.smoothl1 = nn.SmoothL1Loss(reduction='none')

    def forward(self, aed_pred, aed_label):
        num = torch.sum(aed_label[:, 1, :, :]).float()
        # print(aed_label[:, 1, :, :].sum(dim=0, keepdim=True).shape)
        density = aed_pred.pow(2).sum(dim=1, keepdim=True).sqrt()
        l1_loss = aed_label[:, 1:2, :, :] * self.smoothl1(density, aed_label[:, :1, :, :])
        den_loss = 5 * torch.sum(l1_loss) / max(1.0, num)

        diversity = torch.div(aed_pred, density+1e-4) * aed_label[:, 1:2, :, :]
        aver_diversity = diversity.sum(dim=1, keepdim=True) * 1 / 4 * aed_label[:, 1:2, :, :]
        # print(diversity[aver_diversity != 0].shape)
        # print(torch.where(aver_diversity != 0))
        tem_aver = torch.cat((aver_diversity, aver_diversity, aver_diversity, aver_diversity), dim=1)
        pull = 0.0
        push = 0.0
        pull = (diversity - tem_aver).pow(2).sum(dim=1, keepdim=True)
        pull = torch.sum(pull) / max(1.0, num)
 
        mask = aed_label[:, 1, :, :]
        ind = 0
        for batch in zip(aver_diversity):
            aed_data = aver_diversity[ind, 0, :, :]
            tem_mask = mask[ind, :, :]
            aed_data = aed_data * tem_mask
            tem = aed_data[aed_data != 0]
            # print(ind, tem)
            res = aed_data * 0
            for i in range(len(tem)):
                r = 1 - abs(tem - tem[i])
                r[r < 0] = 0
                res[aed_data != 0] = res[aed_data != 0] + r
            res[aed_data != 0] = res[aed_data != 0] - 1
            push += torch.sum(res)
            ind += 1
        # push = push / max(1.0, num * (num - 1))
        '''
        total = len(aver_diversity[0, 0, :, 0]) * len(aver_diversity[0, 0, 0, :])      
        for batch in zip(aver_diversity):
            mask = aed_label[ind:ind + 1, 1:2, :, :]
            mask = mask.squeeze()
            mask = mask.reshape((1, len(aver_diversity[0, 0, :, 0]), len(aver_diversity[0, 0, 0, :])))
            mask = mask == 1
            num_mask = mask.sum(dim=1, keepdim=True).float()

            aed_data = diversity[ind:ind + 1, 0, :, :]
            aed_data = aed_data.squeeze()
            aed_data = aed_data.reshape((1, len(aver_diversity[0, 0, :, 0]), len(aver_diversity[0, 0, 0, :])))

            tem = aed_data.reshape((total, 1))
            tem_mask = mask.reshape((total, 1))

            push_mask = (mask & tem_mask[:, None, :]) == True  
            dist = F.relu(1 - (aed_data - tem[:, None, :]).abs(), inplace=True)
            dist = dist - 1 / (num_mask[:, None, :] + 1e-4)  
            push += dist[push_mask].sum()
            ind += 1
        '''
        return 0.01 * (den_loss + pull + push)
        
