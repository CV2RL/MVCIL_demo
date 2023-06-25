# -*- codeing = utf-8 -*-
"""
作者: WXZ
日期: 2023年06月20日
"""
from utils.dataloader import SinglePoint as dataloader
import numpy as np
from torch.utils.data import DataLoader


class ModelNet40(dataloader):
    def __init__(self, root_dir, scale_aug=False, rot_aug=False, npoint=1024, test_mode=False, num_models=0,
                 num_views=20, normalize=True, train=True):
        super(ModelNet40, self).__init__(root_dir=root_dir, scale_aug=scale_aug, rot_aug=rot_aug, npoint=npoint,
                                         test_mode=test_mode, num_models=num_models, num_views=num_views,
                                         normalize=normalize, train=train)
        from tqdm import tqdm
        import torch
        import utils.provider
        self.normalize = normalize
        self.num_point = npoint
        self.train = train
        # if self.train:
        #     self.root_dir = self.root_dir + 'single_view_modelnet/*/train'
        # else:
        #     self.root_dir = self.root_dir + 'single_view_modelnet/*/test'
        self.points = []
        self.target = []
        self.dataset = dataloader(self.root_dir, npoint=self.num_point, train=self.train)
        if self.train:
            print("Strat loading train data")
            dataLoader = DataLoader(self.dataset, batch_size=200, shuffle=True, num_workers=0)
        else:
            print("Strat loading test data")
            dataLoader = DataLoader(self.dataset, batch_size=100, shuffle=False, num_workers=0)

        for points, target in tqdm(dataLoader):
            points = points.numpy()
            target = target.numpy()
            points = utils.provider.random_point_dropout(points)
            points[:, :, 0:3] = utils.provider.random_scale_point_cloud(points[:, :, 0:3])
            points[:, :, 0:3] = utils.provider.shift_point_cloud(points[:, :, 0:3])
            # points = torch.tensor(points).float()
            self.points = points if self.points == [] else np.concatenate((self.points, points))
            self.target = target if self.target == [] else np.concatenate((self.target, target))
            # self.points = self.points.numpy()
            # self.target = self.target.numpy()

        # for points, target in tqdm(testdataLoader):
        #     points = points.numpy()
        #     target = target.numpy()
        #     points = provider.random_point_dropout(points)
        #     points[:, :, 0:3] = provider.random_scale_point_cloud(points[:, :, 0:3])
        #     points[:, :, 0:3] = provider.shift_point_cloud(points[:, :, 0:3])
        #     # points = torch.tensor(points).float()
        #     self.Testpoints = points if self.TestData == [] else np.concatenate((self.TestData, points))
        #     self.Testtarget = target if self.TestLabels == [] else np.concatenate((self.TestLabels, target))

    def __getitem__(self, index: int):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        point, target = self.points[index], self.target[index]
        # test_point, test_target = self.Testpoints[index], self.Testtarget[index]
        # triain_point, train_target, test_point, test_target = triain_point, train_target, test_point, test_target
        return point, target

        # return triain_point, train_target, test_point, test_target

    def __len__(self) -> int:
        return len(self.data)
