import torch.nn as nn
import torch.nn.functional as F

class network(nn.Module):

    def __init__(self, numclass, feature_extractor):
        super(network, self).__init__()
        self.feature = feature_extractor
        self.fc = nn.Linear(feature_extractor.fc.in_features, numclass, bias=True)

    def forward(self, input):
        x = self.feature(input)
        x = self.fc(x)
        return x

    def Incremental_learning(self, numclass):
        weight = self.fc.weight.data
        bias = self.fc.bias.data
        in_feature = self.fc.in_features
        out_feature = self.fc.out_features

        self.fc = nn.Linear(in_feature, numclass, bias=True)
        self.fc.weight.data[:out_feature] = weight
        self.fc.bias.data[:out_feature] = bias

    def feature_extractor(self, inputs):
        return self.feature(inputs)


class point_network(nn.Module):

    def __init__(self, numclass, feature_extractor):
        super(point_network, self).__init__()
        self.feature = feature_extractor
        self.fc1 = nn.Linear(1024, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.drop1 = nn.Dropout(0.4)
        self.fc2 = nn.Linear(512, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.drop2 = nn.Dropout(0.5)
        self.fc3 = nn.Linear(256, numclass)

    def forward(self, input):
        x = self.feature(input)
        x = self.drop1(F.relu(self.bn1(self.fc1(x))))
        x = self.drop2(F.relu(self.bn2(self.fc2(x))))
        x = self.fc3(x)
        x = F.log_softmax(x, -1)
        return x

    def Incremental_learning(self, numclass):
        weight1 = self.fc1.weight.data
        bias1 = self.fc1.bias.data
        weight2 = self.fc2.weight.data
        bias2 = self.fc2.bias.data
        weight3 = self.fc3.weight.data
        bias3 = self.fc3.bias.data
        in_feature1 = self.fc1.in_features
        out_feature1 = self.fc1.out_features
        in_feature2 = self.fc2.in_features
        out_feature2 = self.fc2.out_features
        in_feature3 = self.fc3.in_features
        out_feature3 = self.fc3.out_features

        self.fc1 = nn.Linear(1024, 512, bias=True)
        self.fc2 = nn.Linear(512, 256, bias=True)
        self.fc3 = nn.Linear(256, numclass, bias=True)
        self.fc1.weight.data[:out_feature1] = weight1
        self.fc1.bias.data[:out_feature1] = bias1
        self.fc2.weight.data[:out_feature2] = weight2
        self.fc2.bias.data[:out_feature2] = bias2
        self.fc3.weight.data[:out_feature3] = weight3
        self.fc3.bias.data[:out_feature3] = bias3

    def feature_extractor(self, inputs):
        return self.feature(inputs)
