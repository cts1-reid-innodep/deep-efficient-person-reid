import torch


def train_collate_fn(batch):
    imgs, pids, _, _, = zip(*batch)
    pids = torch.tensor(pids, dtype=torch.int64)
    return torch.stack(imgs, dim=0), pids


def val_collate_fn(batch):
    imgs, pids, camids, img_paths = zip(*batch)
    return torch.stack(imgs, dim=0), pids, camids, img_paths

# gender, upper, lower
def train_attr_collate_fn(batch):
    imgs, pids, _, _, attr = zip(*batch)
    pids = torch.tensor(pids, dtype=torch.int64)
    attr = torch.tensor(attr, dtype=torch.int64)
    return torch.stack(imgs, dim=0), pids, attr

def val_attr_collate_fn(batch):
    imgs, pids, camids, attr, img_paths = zip(*batch)
    return torch.stack(imgs, dim=0), pids, camids, attr, img_paths