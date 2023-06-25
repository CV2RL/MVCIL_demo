import numpy as np
import glob
import torch.utils.data
import torch
from utils.provider import pc_normalize


class SinglePoint(torch.utils.data.Dataset):
    def __init__(self, root_dir, scale_aug=False, rot_aug=False, npoint=1024, test_mode=False, num_models=0,
                 num_views=20, normalize=False,train = True):
        self.classnames = ['airplane', 'bathtub', 'bed', 'bench', 'bookshelf', 'bottle', 'bowl', 'car', 'chair',
                           'cone', 'cup', 'curtain', 'desk', 'door', 'dresser', 'flower_pot', 'glass_box',
                           'guitar', 'keyboard', 'lamp', 'laptop', 'mantel', 'monitor', 'night_stand',
                           'person', 'piano', 'plant', 'radio', 'range_hood', 'sink', 'sofa', 'stairs',
                           'stool', 'table', 'tent', 'toilet', 'tv_stand', 'vase', 'wardrobe', 'xbox']

        self.root_dir = root_dir
        self.scale_aug = scale_aug
        self.rot_aug = rot_aug
        self.test_mode = test_mode
        self.npoints = npoint
        self.normalize = normalize
        self.train = train
        if self.train:
            self.root_dir = self.root_dir + 'single_view_modelnet/*/train'
        else:
            self.root_dir = self.root_dir + 'single_view_modelnet/*/test'
        set_ = root_dir.split('/')[-1]
        parent_dir = root_dir.rsplit('/', 2)[0]
        self.filepaths = []

        for i in range(len(self.classnames)):
            all_files = sorted(glob.glob(parent_dir + '/' + self.classnames[i] + '/' + set_ + '/*.xyz'))
            self.filepaths.extend(all_files)

    def __len__(self):
        # print(len(self.filepaths))
        return len(self.filepaths)

    def __getitem__(self, idx):
        path = self.filepaths[idx]
        class_name = path.split('/')[-2]
        class_id = self.classnames.index(class_name)
        point_set = np.loadtxt(self.filepaths[idx])
        if self.normalize:
            point_set[:, :3] = pc_normalize(point_set[:, :3])
        return (point_set, class_id)
