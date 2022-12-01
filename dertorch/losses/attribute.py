import torch
from torch import nn
from torch.autograd import Variable


def euclidean_dist(x, y):
    """
    Args:
      x: pytorch Variable, with shape [m, d]
      y: pytorch Variable, with shape [n, d]
    Returns:
      dist: pytorch Variable, with shape [m, n]
    """
    m, n = x.size(0), y.size(0)
    xx = torch.pow(x, 2).sum(1, keepdim=True).expand(m, n)
    yy = torch.pow(y, 2).sum(1, keepdim=True).expand(n, m).t()
    dist = xx + yy
    dist.addmm_(x, y.t(), beta=1, alpha=-2)
    dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability
    return dist

class TripletLossAttrWeightes(nn.Module):
    def __init__(self, margin=None, dis_type="euclid"):
        super(TripletLossAttrWeightes, self).__init__()
        self.margin = margin
        self.ranking_loss = nn.MarginRankingLoss(margin=margin)
        self.dis_type = dis_type

    def forward(self, inputs, targets, weights_vector):
        """
        Args:
            inputs: feature matrix with shape (batch_size, feat_dim)
            targets: ground truth labels with shape (num_classes)
        """
        n = inputs.size(0)
        # Compute pairwise distance, replace by the official when merged
        if self.dis_type == "cosine":
            inputs = inputs / (inputs ** 2).sum(dim=1, keepdim=True).sqrt()
            dist = 1 - inputs.mm(inputs.t())
        elif self.dis_type == "euclid":
            dist = euclidean_dist(inputs, inputs)
        # w_dis = dist
        w_dis = euclidean_dist(weights_vector[0], weights_vector[0])
        # For each anchor, find the hardest positive and negative
        mask = targets.expand(n, n).eq(targets.expand(n, n).t())
        dist_ap, dist_an = [], []
        for i in range(n):
            dist_ap.append(dist[i][mask[i]][w_dis[i][mask[i]].argmax()].reshape(1))
            dist_an.append(dist[i][mask[i]][w_dis[i][mask[i]].argmin()].reshape(1))
        dist_ap = torch.cat(dist_ap, -1)
        dist_an = torch.cat(dist_an, -1)
        # Compute ranking hinge loss
        y = dist_an.data.new()
        y.resize_as_(dist_an.data)
        y.fill_(1)
        y = Variable(y)
        loss = self.ranking_loss(dist_an, dist_ap, y)
        return loss