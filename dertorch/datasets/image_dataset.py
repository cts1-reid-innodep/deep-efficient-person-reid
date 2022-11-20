import os.path as osp
from PIL import Image
from torch.utils.data import Dataset
import torch


def read_image(img_path):
    """Keep reading image until succeed.
    This can avoid IOError incurred by heavy IO process."""
    got_img = False
    if not osp.exists(img_path):
        raise IOError("{} does not exist".format(img_path))
    while not got_img:
        try:
            img = Image.open(img_path).convert('RGB')
            got_img = True
        except IOError:
            print("IOError incurred when reading '{}'. Will redo. Don't worry. Just chill.".format(img_path))
            pass
    return img


class ImageDataset(Dataset):
    """Image Person ReID Dataset"""

    def __init__(self, dataset, transform=None, attr_lens=[]):
        self.dataset = dataset # = aihubdataset.train
        self.transform = transform # = train_transform
        self.attr_lens = attr_lens

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        img_path, pid, camid, attr = self.dataset[index]
        attributes = []
        img = read_image(img_path)

        if self.transform is not None:
            img = self.transform(img)

        if len(self.attr_lens) != 0:
            attribute = []
            for i, a in enumerate(attrs):
                if i < len(self.attr_lens):
                    attr = [1 if _ == a else 0 for _ in range(self.attr_lens[i])]
                    attribute.extend(attr)
                else:
                    attr = [1 if _ == a else 0 for _ in range(self.attr_lens[i-len(self.attr_lens)])]
                    attribute.extend(attr)
                if i == len(self.attr_lens) - 1:
                    attribute = torch.Tensor(attribute)
                    attributes.append(attribute)
                    attribute = []

        return img, pid, camid, img_path, attributes