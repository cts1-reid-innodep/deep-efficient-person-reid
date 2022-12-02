import glob
import re
import os.path as osp
import json

from .base import BaseImageDataset
from scipy import io

class Innodep(BaseImageDataset):
    """
    Dataset statistics:
    # identities: 330
    # images: 10064 (train) + 822 (query) + 6368 (gallery)
    """
    columns = ["gender", "upperclothes", "lowerclothes"]
    attr_lens = [2, 2, 2]

    def __init__(self, root="/home/snu1/innodep_reid", verbose=True, **kwargs):
        super(Innodep, self).__init__()
        self.dataset_dir = root
        self.train_dir = osp.join(self.dataset_dir, 'train')
        self.query_dir = osp.join(self.dataset_dir, 'query')
        self.gallery_dir = osp.join(self.dataset_dir, 'gallery')

        self._check_before_run()

        train = self._process_dir(self.train_dir, relabel=True, attr=True)
        query = self._process_dir(self.query_dir, relabel=False, attr=False)
        gallery = self._process_dir(self.gallery_dir, relabel=False, attr=False)

        if verbose:
            print("=> Innodep loaded")
            self.print_dataset_statistics(train, query, gallery)

        self.train = train
        self.query = query
        self.gallery = gallery

        self.num_train_pids, self.num_train_imgs, self.num_train_cams = self.get_imagedata_info(
            self.train)
        self.num_query_pids, self.num_query_imgs, self.num_query_cams = self.get_imagedata_info(
            self.query)
        self.num_gallery_pids, self.num_gallery_imgs, self.num_gallery_cams = self.get_imagedata_info(
            self.gallery)

    def _check_before_run(self):
        """Check if all files are available before going deeper"""
        if not osp.exists(self.dataset_dir):
            raise RuntimeError(
                "'{}' is not available".format(self.dataset_dir))
        if not osp.exists(self.train_dir):
            raise RuntimeError("'{}' is not available".format(self.train_dir))
        if not osp.exists(self.query_dir):
            raise RuntimeError("'{}' is not available".format(self.query_dir))
        if not osp.exists(self.gallery_dir):
            raise RuntimeError(
                "'{}' is not available".format(self.gallery_dir))

    def _process_dir(self, dir_path, relabel=False, attr=False):
        cam_paths = glob.glob(osp.join(dir_path, '*'))
        pattern = re.compile(r'\d_(\d)_([\d]+)_(\d)_(\d)')

        dataset = []
        for cam_path in cam_paths:
            per, cam1, cam2, chk = map(int, pattern.search(cam_path).groups())
            camid = 2 * (cam1 - 1) + cam2
            pid_paths = glob.glob(osp.join(cam_path, '*'))
            for pid_path in pid_paths:
                img_paths = glob.glob(osp.join(pid_path, '*.jpg'))
                for img_path in img_paths:
                    attr_path = img_path.replace('jpg', 'json')
                    with open(attr_path, 'r') as f:
                        json_data = json.load(f)
                        gender = json_data['object']['']['gender']
                        tops_type = json_data['object']['']['tops_type']
                        bottoms_type = json_data['object']['']['bottoms_type']
                    if (gender != 1 and gender != 2): continue
                    if (per == 1 and chk == 1): pid = json_data['id']
                    elif (per == 1 and chk == 2): pid = 55 + json_data['id']
                    elif ((per == 2 or per == 3) and chk == 2):
                        if (json_data['id'] == 52 and per == 3): pid = 165
                        else: pid = 112 + json_data['id']
                    elif (per == 4 and chk == 1): pid = 165 + json_data['id']
                    elif (per == 5 and chk == 1): pid = 220 + json_data['id']
                    else: pid = 275 + json_data['id']
                    if attr:
                        attribute = [gender, tops_type, bottoms_type]
                    else:
                        attribute = []

                    dataset.append((img_path, pid-1, camid-1, attribute))

        return dataset