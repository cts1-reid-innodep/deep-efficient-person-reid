import torch
from torch import nn
from torch.autograd import Variable


def normalize(x, axis=-1):
    """Normalizing to unit length along the specified dimension.
    Args:
      x: pytorch Variable
    Returns:
      x: pytorch Variable, same shape as input
    """
    x = 1. * x / (torch.norm(x, 2, axis, keepdim=True).expand_as(x) + 1e-12)
    return x


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


# class TripletLoss(object):
#     """Modified from Tong Xiao's open-reid (https://github.com/Cysu/open-reid).
#     Related Triplet Loss theory can be found in paper 'In Defense of the Triplet
#     Loss for Person Re-Identification'."""

#     def __init__(self, margin=None):

#         self.margin = margin
#         if margin is not None:
#             self.ranking_loss = nn.MarginRankingLoss(margin=margin)
#         else:
#             self.ranking_loss = nn.SoftMarginLoss()

#     def __call__(self, global_feat, labels, normalize_feature=False):
#         n = global_feat.size(0)

#         # Compute pairwise distance, replace by the official when merged
#         dist = torch.pow(global_feat, 2).sum(dim=1, keepdim=True).expand(n, n)
#         dist = dist + dist.t()
#         dist.addmm_(global_feat, global_feat.t(), beta=1, alpha=-2)
#         dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability

#         # For each anchor, find the hardest positive and negative
#         mask = labels.expand(n, n).eq(labels.expand(n, n).t())
#         dist_ap, dist_an = [], []
#         for i in range(n):
#             dist_ap.append(dist[i][mask[i]].max())
#             dist_an.append(dist[i][mask[i] == 0].min())

#         dist_ap = torch.stack(dist_ap)
#         dist_an = torch.stack(dist_an)

#         # Compute ranking hinge loss
#         y = dist_an.data.new()
#         y.resize_as_(dist_an.data)
#         y.fill_(1)
#         y = Variable(y)
#         loss = self.ranking_loss(dist_an, dist_ap, y)
#         prec = (dist_an.data > dist_ap.data).sum() * 1. / y.size(0)

#         return loss, prec

def hard_example_mining(dist_mat, labels, return_inds=False, HSoft=False):
    """For each anchor, find the hardest positive and negative sample.
    Args:
      dist_mat: pytorch Variable, pair wise distance between samples, shape [N, N]
      labels: pytorch LongTensor, with shape [N]
      return_inds: whether to return the indices. Save time if `False`(?)
    Returns:
      dist_ap: pytorch Variable, distance(anchor, positive); shape [N]
      dist_an: pytorch Variable, distance(anchor, negative); shape [N]
      p_inds: pytorch LongTensor, with shape [N];
        indices of selected hard positive samples; 0 <= p_inds[i] <= N - 1
      n_inds: pytorch LongTensor, with shape [N];
        indices of selected hard negative samples; 0 <= n_inds[i] <= N - 1
    NOTE: Only consider the case in which all labels have same num of samples,
      thus we can cope with all anchors in parallel.
    """

    assert len(dist_mat.size()) == 2
    assert dist_mat.size(0) == dist_mat.size(1)
    N = dist_mat.size(0)

    # shape [N, N]

    is_pos = labels.expand(N, N).eq(labels.expand(N, N).t())
    is_neg = labels.expand(N, N).ne(labels.expand(N, N).t())

    # `dist_ap` means distance(anchor, positive)
    # both `dist_ap` and `relative_p_inds` with shape [N, 1]
    if HSoft == True:
        num_p = is_pos.sum(dim=1).float() - 0.5
        dist_ap = (dist_mat * is_pos.float()).sum(dim=1) / num_p
    else:
        dist_ap, relative_p_inds = torch.max(
            dist_mat * is_pos.float(), 1, keepdim=True)
        dist_ap = dist_ap.squeeze(1)

    dist_an, relative_n_inds = torch.min(
        dist_mat * is_neg.float(), 1, keepdim=True)
    dist_an = dist_an.squeeze(1)

    if return_inds:
        # shape [N, N]
        ind = (labels.new().resize_as_(labels)
               .copy_(torch.arange(0, N).long())
               .unsqueeze(0).expand(N, N))
        # shape [N, 1]
        p_inds = torch.gather(
            ind[is_pos].contiguous().view(N, -1), 1, relative_p_inds.data)
        n_inds = torch.gather(
            ind[is_neg].contiguous().view(N, -1), 1, relative_n_inds.data)
        # shape [N]
        p_inds = p_inds.squeeze(1)
        n_inds = n_inds.squeeze(1)
        return dist_ap, dist_an, p_inds, n_inds

    return dist_ap, dist_an


class TripletLoss(object):
    """Modified from Tong Xiao's open-reid (https://github.com/Cysu/open-reid).
    Related Triplet Loss theory can be found in paper 'In Defense of the Triplet
    Loss for Person Re-Identification'."""

    def __init__(self, margin=None):
        self.margin = margin
        if margin is not None:
            self.ranking_loss = nn.MarginRankingLoss(margin=margin)
        else:
            self.ranking_loss = nn.SoftMarginLoss()

    def __call__(self, global_feat, labels, normalize_feature=True, HSoft=False):
        if normalize_feature:
            global_feat = normalize(global_feat, axis=-1)
        dist_mat = euclidean_dist(global_feat, global_feat)
        dist_ap, dist_an = hard_example_mining(
            dist_mat, labels, HSoft=HSoft)
        y = dist_an.new().resize_as_(dist_an).fill_(1)
        if self.margin is not None:
            loss = self.ranking_loss(dist_an, dist_ap, y)
        else:
            loss = self.ranking_loss(dist_an - dist_ap, y)
        return loss, dist_ap, dist_an

class CrossEntropyLabelSmooth(nn.Module):
    """Cross entropy loss with label smoothing regularizer.
    Reference:
    Szegedy et al. Rethinking the Inception Architecture for Computer Vision. CVPR 2016.
    Equation: y = (1 - epsilon) * y + epsilon / K.
    Args:
        num_classes (int): number of classes.
        epsilon (float): weight.
    """

    def __init__(self, num_classes, epsilon=0.1, use_gpu=True):
        super(CrossEntropyLabelSmooth, self).__init__()
        self.num_classes = num_classes
        self.epsilon = epsilon
        self.use_gpu = use_gpu
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, inputs, targets):
        """
        Args:
            inputs: prediction matrix (before softmax) with shape (batch_size, num_classes)
            targets: ground truth labels with shape (num_classes)
        """
        log_probs = self.logsoftmax(inputs)
        targets = torch.zeros(log_probs.size()).scatter_(
            1, targets.unsqueeze(1).data.cpu(), 1)
        if self.use_gpu:
            targets = targets.cuda()
        targets = (1 - self.epsilon) * targets + \
            self.epsilon / self.num_classes
        loss = (- targets * log_probs).mean(0).sum()
        return loss
