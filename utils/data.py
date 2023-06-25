# -*- codeing = utf-8 -*-
"""
作者: WXZ
日期: 2023年06月19日
"""
from torchvision import transforms
from utils.dataloader import SinglePoint
from torch.utils.data import DataLoader
import numpy as np
import utils.iCIFAR100
import utils.iModelNet


class iData(object):
    train_transform = []
    test_transform = []
    classify_transform = []


class iCIFAR(iData):
    def __init__(self, dataset_name, init_class, increment, batchsize, exemplar_set):
        super(iData, self).__init__()
        self.init_class = init_class
        self.increment = increment
        self.batchsize = batchsize
        self.exemplar_set = exemplar_set
        self.transform = transforms.Compose([  # transforms.Resize(img_size),
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))])

    train_transform = transforms.Compose([  # transforms.Resize(img_size),
        transforms.RandomCrop((32, 32), padding=4),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.24705882352941178),
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))])

    test_transform = transforms.Compose([  # transforms.Resize(img_size),
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))])

    classify_transform = transforms.Compose([transforms.RandomHorizontalFlip(p=1.),
                                             # transforms.Resize(img_size),
                                             transforms.ToTensor(),
                                             transforms.Normalize((0.5071, 0.4867, 0.4408),
                                                                  (0.2675, 0.2565, 0.2761))])

    def download_data(self):
        classes = [self.init_class - self.increment, self.init_class]
        self.train_dataset = utils.iCIFAR100.iCIFAR100('dataset', transform=self.train_transform, download=False)
        self.test_dataset = utils.iCIFAR100.iCIFAR100('dataset', test_transform=self.test_transform, train=False,
                                                      download=False)
        self.train_loader, self.test_loader = self._get_train_and_test_dataloader(classes)

    def _get_train_and_test_dataloader(self, classes):
        self.train_dataset.getTrainData(classes, self.exemplar_set)
        self.test_dataset.getTestData(classes)
        train_loader = DataLoader(dataset=self.train_dataset,
                                  shuffle=True,
                                  batch_size=self.batchsize)

        test_loader = DataLoader(dataset=self.test_dataset,
                                 shuffle=True,
                                 batch_size=self.batchsize)

        return train_loader, test_loader


class imodelnet(iData):
    def __init__(self, root_dir, init_class, increment, batchsize, exemplar_set, num_point):
        super(iData, self).__init__()
        self.root_dir = root_dir
        self.init_class = init_class
        self.increment = increment
        self.batchsize = batchsize
        self.exemplar_set = exemplar_set
        self.num_point = num_point

    def download_data(self):
        classes = [self.init_class - self.increment, self.init_class]
        self.train_dataset = utils.iModelNet.iModelNet40(root_dir=self.root_dir, npoint=self.num_point, train=True)
        self.test_dataset = utils.iModelNet.iModelNet40(root_dir=self.root_dir, npoint=self.num_point, train=False)
        self.train_loader, self.test_loader = self._get_train_and_test_dataloader(classes)

    def _get_train_and_test_dataloader(self, classes):
        self.train_dataset.getTrainData(classes, self.exemplar_set)
        self.test_dataset.getTestData(classes)
        train_loader = DataLoader(dataset=self.train_dataset,
                                  shuffle=True,
                                  batch_size=self.batchsize)

        test_loader = DataLoader(dataset=self.test_dataset,
                                 shuffle=True,
                                 batch_size=self.batchsize)

        return train_loader, test_loader
