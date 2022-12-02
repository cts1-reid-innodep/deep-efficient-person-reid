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

    def __init__(self, root='market1501', verbose=True, **kwargs):
        super(Market1501, self).__init__()
        self.dataset_dir = root
        self.train_dir = osp.join(self.dataset_dir, 'bounding_box_train')
        self.query_dir = osp.join(self.dataset_dir, 'query')
        self.gallery_dir = osp.join(self.dataset_dir, 'bounding_box_test')
        self.attributes_dir = "/home/snu1/datasets/market1501/attribute/market_attribute.mat"

        self._check_before_run()

        train = self._process_dir(self.train_dir, relabel=True, attr=True)
        query = self._process_dir(self.query_dir, relabel=False)
        gallery = self._process_dir(self.gallery_dir, relabel=False)

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

    def _process_dir(self, dir_path, relabel=False, attr = False):
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
                fl = f['market_attribute'][0][0][test_train][0][0]
                if test_train == 0:
                    continue
                else:
                    id_list_name = 'train_person_id'
                    group_name = 'train_attribute'   
                for person_id in range(len(fl[26][0])):
                    gen = fl['gender'][0][person_id] - 1
                    
                    uptype = ['upblack', 'upwhite', 'upred', 'uppurple', 'upyellow', 'upgray', 'upblue', 'upgreen']
                    downtype = ['downblack', 'downwhite', 'downpink', 'downpurple', 'downyellow', 'downgray', 'downblue', 'downgreen', 'downbrown']
                    
                    if fl['upblack'][0][person_id] == 2 :
                        upper = 0
                    elif fl['upwhite'][0][person_id] == 2 :
                        upper = 1
                    elif fl['upred'][0][person_id] == 2 :
                        upper = 2
                    elif fl['uppurple'][0][person_id] == 2 :
                        upper = 3
                    elif fl['upyellow'][0][person_id] == 2 :
                        upper = 4
                    elif fl['upgray'][0][person_id] == 2 :
                        upper = 5
                    elif fl['upblue'][0][person_id] == 2 :
                        upper = 6
                    elif fl['upgreen'][0][person_id] == 2 :
                        upper = 7

                    if fl['downblack'][0][person_id] == 2 :
                        lower = 0
                    elif fl['downwhite'][0][person_id] == 2 :
                        lower = 1
                    elif fl['downpink'][0][person_id] == 2 :
                        lower = 2
                    elif fl['downpurple'][0][person_id] == 2 :
                        lower = 3
                    elif fl['downyellow'][0][person_id] == 2 :
                        lower = 4
                    elif fl['downgray'][0][person_id] == 2 :
                        lower = 5
                    elif fl['downblue'][0][person_id] == 2 :
                        lower = 6
                    elif fl['downgreen'][0][person_id] == 2 :
                        lower = 7
                    elif fl['downbrown'][0][person_id] == 2 :
                        lower = 8

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
