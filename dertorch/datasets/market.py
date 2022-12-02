import glob
import re
import os.path as osp

from .base import BaseImageDataset
from scipy import io

class Market1501(BaseImageDataset):
    """
    Dataset statistics:
    # identities: 1501 (+1 for background)
    # images: 12936 (train) + 3368 (query) + 15913 (gallery)
    """
    columns = ["gender", "upperclothes", "lowerclothes"]
    attr_lens = [2, 2, 4]

    def __init__(self, root="/home/snu1/datasets/market1501", verbose=True, **kwargs):
        super(Market1501, self).__init__()
        self.dataset_dir = root
        self.train_dir = osp.join(self.dataset_dir, 'bounding_box_train')
        self.query_dir = osp.join(self.dataset_dir, 'query')
        self.gallery_dir = osp.join(self.dataset_dir, 'bounding_box_test')
        self.attributes_dir = "/home/snu1/datasets/market1501/attribute/market_attribute.mat"

        self._check_before_run()

        train = self._process_dir(self.train_dir, relabel=True, attr=True)
        query = self._process_dir(self.query_dir, relabel=False, attr=False)
        gallery = self._process_dir(self.gallery_dir, relabel=False, attr=False)

        if verbose:
            print("=> Market1501 loaded")
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
        img_paths = glob.glob(osp.join(dir_path, '*.jpg'))
        pattern = re.compile(r'([-\d]+)_c(\d)')

        pid_container = set()
        for img_path in img_paths:
            pid, _ = map(int, pattern.search(img_path).groups())
            if pid == -1:
                continue  # junk images are just ignored
            pid_container.add(pid)
        pid2label = {pid: label for label, pid in enumerate(pid_container)}

        attr_container = dict()
        if attr:
            f = io.loadmat(self.attributes_dir)
            for test_train in range(len(f['market_attribute'][0][0])):
                if test_train == 0:
                    continue
                else:
                    id_list_name = 'train_person_id'
                    group_name = 'train_attribute'   
                for person_id in range(len(f['market_attribute'][0][0][test_train][0][0][26][0])):
                    gen = f['market_attribute'][0][0][test_train][0][0][26][0][person_id] - 1
                    upper = f['market_attribute'][0][0][test_train][0][0][23][0][person_id] - 1
                    if f['market_attribute'][0][0][test_train][0][0][21][0][person_id] == 1 and f['market_attribute'][0][0][test_train][0][0][22][0][person_id] == 1:
                        lower = 0
                    elif f['market_attribute'][0][0][test_train][0][0][21][0][person_id] == 1 and f['market_attribute'][0][0][test_train][0][0][22][0][person_id] == 2:
                        lower = 1
                    elif f['market_attribute'][0][0][test_train][0][0][21][0][person_id] == 2 and f['market_attribute'][0][0][test_train][0][0][22][0][person_id] == 1:
                        lower = 2
                    elif f['market_attribute'][0][0][test_train][0][0][21][0][person_id] == 2 and f['market_attribute'][0][0][test_train][0][0][22][0][person_id] == 2:
                        lower = 3
                    att = [gen, upper, lower]
                    attr_container[person_id] = att
                    # print("pid: {} attr: {}".format(person_id, attr_container[person_id]))

        dataset = []
        for img_path in img_paths:
            pid, camid = map(int, pattern.search(img_path).groups())
            if pid == -1:
                continue  # junk images are just ignored
            assert 0 <= pid <= 1501  # pid == 0 means background
            assert 1 <= camid <= 6
            camid -= 1  # index starts from 0
            attribute = []
            if relabel:
                pid = pid2label[pid]
                attribute = attr_container[pid]
            dataset.append((img_path, pid, camid, attribute))

        return dataset


class Market1501_Oneshot(BaseImageDataset):
    """
    Market1501
    Reference:
    Zheng et al. Scalable Person Re-identification: A Benchmark. ICCV 2015.
    URL: http://www.liangzheng.org/Project/project_reid.html
    Dataset statistics:
    # identities: 1501 (+1 for background)
    # images: 12936 (train) + 3368 (query) + 15913 (gallery)
    """
    columns = ["gender", "upperclothes", "lowerclothes"]
    attr_lens = [2, 2, 4]

    def __init__(self, root="/home/snu1/datasets/market1501", verbose=True, **kwargs):
        super(Market1501_Oneshot, self).__init__()
        self.dataset_dir = root
        self.train_dir = osp.join(self.dataset_dir, 'bounding_box_train_query_three_shot') #bounding_box_train_query_one_random
        self.query_dir = osp.join(self.dataset_dir, 'query')
        self.gallery_dir = osp.join(self.dataset_dir, 'bounding_box_test')
        self.attributes_dir = "/home/snu1/datasets/market1501/attribute/market_attribute.mat"

        self._check_before_run()

        train = self._process_dir(self.train_dir, relabel=True, attr=True)
        query = self._process_dir(self.query_dir, relabel=False, attr=False)
        gallery = self._process_dir(self.gallery_dir, relabel=False, attr=False)

        if verbose:
            print("=> Market1501_Oneshot Loaded")
            self.print_dataset_statistics(train, query, gallery)

        self.train = train
        self.query = query
        self.gallery = gallery

        self.num_train_pids, self.num_train_imgs, self.num_train_cams = self.get_imagedata_info(self.train)
        self.num_query_pids, self.num_query_imgs, self.num_query_cams = self.get_imagedata_info(self.query)
        self.num_gallery_pids, self.num_gallery_imgs, self.num_gallery_cams = self.get_imagedata_info(self.gallery)

    def _check_before_run(self):
        """Check if all files are available before going deeper"""
        if not osp.exists(self.dataset_dir):
            raise RuntimeError("'{}' is not available".format(self.dataset_dir))
        if not osp.exists(self.train_dir):
            raise RuntimeError("'{}' is not available".format(self.train_dir))
        if not osp.exists(self.query_dir):
            raise RuntimeError("'{}' is not available".format(self.query_dir))
        if not osp.exists(self.gallery_dir):
            raise RuntimeError("'{}' is not available".format(self.gallery_dir))

    def _process_dir(self, dir_path, relabel=False, attr=False):
        img_paths = glob.glob(osp.join(dir_path, '*.jpg'))
        pattern = re.compile(r'([-\d]+)_c(\d)')

        pid_container = set()
        for img_path in img_paths:
            pid, _ = map(int, pattern.search(img_path).groups())
            if pid == -1: continue  # junk images are just ignored
            pid_container.add(pid)
        pid2label = {pid: label for label, pid in enumerate(pid_container)}

        attr_container = dict()
        if attr:
            f = io.loadmat(self.attributes_dir)
            for test_train in range(len(f['market_attribute'][0][0])):
                if test_train == 0:
                    continue
                else:
                    id_list_name = 'train_person_id'
                    group_name = 'train_attribute'   
                for person_id in range(len(f['market_attribute'][0][0][test_train][0][0][26][0])):
                    gen = f['market_attribute'][0][0][test_train][0][0][26][0][person_id] - 1
                    upper = f['market_attribute'][0][0][test_train][0][0][23][0][person_id] - 1
                    if f['market_attribute'][0][0][test_train][0][0][21][0][person_id] == 1 and f['market_attribute'][0][0][test_train][0][0][22][0][person_id] == 1:
                        lower = 0
                    elif f['market_attribute'][0][0][test_train][0][0][21][0][person_id] == 1 and f['market_attribute'][0][0][test_train][0][0][22][0][person_id] == 2:
                        lower = 1
                    elif f['market_attribute'][0][0][test_train][0][0][21][0][person_id] == 2 and f['market_attribute'][0][0][test_train][0][0][22][0][person_id] == 1:
                        lower = 2
                    elif f['market_attribute'][0][0][test_train][0][0][21][0][person_id] == 2 and f['market_attribute'][0][0][test_train][0][0][22][0][person_id] == 2:
                        lower = 3
                    att = [gen, upper, lower]
                    attr_container[person_id] = att
                    # print("pid: {} attr: {}".format(person_id, attr_container[person_id]))


        dataset = []
        for img_path in img_paths:
            pid, camid = map(int, pattern.search(img_path).groups())
            if pid == -1: continue  # junk images are just ignored
            assert 0 <= pid <= 1501  # pid == 0 means background
            assert 1 <= camid <= 6
            camid -= 1  # index starts from 0
            attribute = []
            if relabel: 
                pid = pid2label[pid]
                attribute = attr_container[pid]
            dataset.append((img_path, pid, camid, attribute))

        return dataset