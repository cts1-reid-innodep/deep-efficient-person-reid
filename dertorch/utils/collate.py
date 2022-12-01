import torch


def train_collate_fn(batch):
    imgs, pids, _, img_paths, attr = zip(*batch)
    pids = torch.tensor(pids, dtype=torch.int64)
    return torch.stack(imgs, dim=0), pids, img_paths, attr


def val_collate_fn(batch):
    imgs, pids, camids, img_paths, _ = zip(*batch)
    return torch.stack(imgs, dim=0), pids, camids, img_paths
