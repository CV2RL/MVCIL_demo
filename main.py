from models.iCaRL import iCaRLmodel
from convs.ResNet import resnet18_cbam
from convs.POINTNET_MSG_CLS import PointView_GCN
import utils.data as data
import utils.provider as provider
import utils.iCIFAR100
from torch.utils.data import DataLoader
import torch

exemplar_set = []
init_class = 10
# feature_extractor = resnet18_cbam()
feature_extractor = PointView_GCN()
batch_size = 128
increment = 10
memory_size = 2000
epochs = 100
learning_rate = 2.0
# # CIFAR
# img_size = 32
# model_str = "ResNet"
# dataset_name = "iCIFAR100"
# modelnet
root_dir = "dataset/"
num_point = 1024
model_str = "PointView_GCN"
dataset_name = "modelnet"

if model_str == "ResNet":
    if "iCIFAR100" in dataset_name:
        idata = data.iCIFAR(dataset_name, init_class, increment, batch_size, exemplar_set)
        idata.download_data()
        train_dataset = idata.train_dataset
        test_dataset = idata.test_dataset
        train_loader = idata.train_loader
        test_loader = idata.test_loader

elif model_str == "PointView_GCN":
    if "modelnet" in dataset_name:
        idata = data.imodelnet(root_dir, init_class, increment, batch_size, exemplar_set, num_point)
        idata.download_data()
        train_dataset = idata.train_dataset
        test_dataset = idata.test_dataset
        train_loader = idata.train_loader
        test_loader = idata.test_loader

# if model_str == "PointView-GCN":
#     if "modelnet" in dataset_name:


model = iCaRLmodel(dataset_name, init_class, feature_extractor, batch_size, increment, memory_size, epochs,
                   learning_rate, train_dataset, test_dataset, train_loader, test_loader)
# model.model.load_state_dict(torch.load('model/ownTry_accuracy:84.000_KNN_accuracy:84.000_increment:10_net.pkl'))

for i in range(4):
    model.beforeTrain()
    accuracy = model.train()
    model.afterTrain(accuracy)
