# -*- codeing = utf-8 -*-
"""
作者: WXZ
日期: 2023年06月21日
"""
from utils.modelnet import ModelNet40 as modelnet
import numpy as np


class iModelNet40(modelnet):
    def __init__(self, root_dir, scale_aug=False, rot_aug=False, npoint=1024, test_mode=False, num_models=0,
                 num_views=20, normalize=True, train=True):
        super(iModelNet40, self).__init__(root_dir=root_dir, scale_aug=scale_aug, rot_aug=rot_aug, npoint=npoint,
                                         test_mode=test_mode, num_models=num_models, num_views=num_views,
                                         normalize=normalize, train=train)
        self.root_dir = root_dir
        self.scale_aug = scale_aug
        self.rot_aug = rot_aug
        self.test_mode = test_mode
        self.npoints = npoint
        self.normalize = normalize
        self.train = train
        self.TrainData = []
        self.TrainLabels = []
        self.TestData = []
        self.TestLabels = []

    def concatenate(self, datas, labels):
        con_data = datas[0]
        con_label = labels[0]
        for i in range(1, len(datas)):
            con_data = np.concatenate((con_data, datas[i]), axis=0)
            con_label = np.concatenate((con_label, labels[i]), axis=0)
        return con_data, con_label

    def getTestData(self,classes):
        datas, labels = [], []
        for label in range(classes[0], classes[1]):
            data = self.points[np.array(self.target) == label]
            datas.append(data)
            labels.append(np.full((data.shape[0]), label))
        datas, labels = self.concatenate(datas, labels)
        self.TestData = datas if self.TestData == [] else np.concatenate((self.TestData, datas), axis=0)
        self.TestLabels = labels if self.TestLabels == [] else np.concatenate((self.TestLabels, labels), axis=0)
        print("the size of test set is %s" % (str(self.TestData.shape)))
        print("the size of test label is %s" % str(self.TestLabels.shape))

    def getTrainData(self, classes, exemplar_set):

        datas, labels = [], []
        if len(exemplar_set) != 0:
            datas = [exemplar for exemplar in exemplar_set]
            length = len(datas[0])
            labels = [np.full((length), label) for label in range(len(exemplar_set))]

        for label in range(classes[0], classes[1]):
            data = self.points[np.array(self.target) == label]
            datas.append(data)
            labels.append(np.full((data.shape[0]), label))
        self.TrainData, self.TrainLabels = self.concatenate(datas, labels)
        print("the size of train set is %s" % (str(self.TrainData.shape)))
        print("the size of train label is %s" % str(self.TrainLabels.shape))

    def getTrainItem(self, index):
        img, target = self.TrainData[index], self.TrainLabels[index]

        return index, img, target

    def getTestItem(self, index):
        img, target = self.TestData[index], self.TestLabels[index]

        return index, img, target

    def __getitem__(self, index):
        if self.TrainData != []:
            return self.getTrainItem(index)
        elif self.TestData != []:
            return self.getTestItem(index)

    def __len__(self):
        if self.TrainData != []:
            return len(self.TrainData)
        elif self.TestData != []:
            return len(self.TestData)

    def get_image_class(self, label):
        return self.data[np.array(self.targets) == label]
