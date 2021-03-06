# -*- coding: utf-8 -*-
"""
Created on Sat Apr 20 11:30:29 2019

@author: ragga
Joint classifier using both audio and video
"""

from __future__ import print_function, division
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision as tv
from torchvision import transforms
import torch.optim as optim
import numpy as np
import os
from torch.utils.data import Dataset



class MyDataset(Dataset):
    def __init__(self, dataset1, dataset2):
        self.dataset1 = dataset1 # datasets should be sorted!
        self.dataset2 = dataset2

    def __getitem__(self, index):
        x2 = self.dataset2[index]
        x1 = self.dataset1[index]
        return x1, x2

    def __len__(self):
        return min(len(self.dataset1), len(self.dataset2)) # assuming both datasets have same length


def npy_loader(path):
    sample = torch.from_numpy(np.load(path))
    return sample


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.img_conv1 = nn.Conv2d(3, 16, 5)
        self.aud_conv1 = nn.Conv1d(1, 16, 3)
        self.drop1 = nn.Dropout(0.5)
        self.img_pool = nn.MaxPool2d(4, 4)
        self.aud_pool = nn.MaxPool1d(2)
        self.img_conv2 = nn.Conv2d(16, 32, 3)
        self.img_pool2 = nn.MaxPool2d(2, 2)
        self.aud_conv2 = nn.Conv1d(16, 32, 3)
        # self.drop2 = nn.Dropout(0.1)
        self.fc1 = nn.Linear(32*(62*30+17), 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 8)
        
    def forward(self, x_img, x_aud):
        x_img = self.img_pool(F.relu(self.img_conv1(x_img)))
        x_img = self.img_pool2(F.relu(self.img_conv2(x_img)))
        x_aud = self.aud_pool(F.relu(self.aud_conv1(x_aud)))
        x_aud = F.relu(self.aud_conv2(x_aud))
        # Note that simple concatination in this manner might not be the
        # best thing to do since one of the features might dominate the 
        # other one. Hence, we can do 
        #   1) Try with equal concatination
        #   2) Audio dominant
        #   3) Video dominant
        #   4) Make the values a hyperparameter
        #   5) Formulate a method to learn this composition ratio

        x_img = x_img.view(-1, 32*30*62)
        x_aud = x_aud.view(-1, 32*17)
        x = torch.cat([x_img, x_aud], dim=1)
        x = self.drop1(x)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


def train(loader, criterion, net, device, optimizer):
    train_loss = []
    val_loss = []
    val_accuracy = []
    train_accuracy = []

    for epoch in range (25):
        running_loss = 0.0
        t_loss = 0
        total = 0
        correct = 0 
        for i, data in enumerate(loader['train']):
            # get the inputs
            img_data, aud_data = data
            img_inputs, img_labels = img_data
            print(img_inputs.shape)
            img_inputs, img_labels = img_inputs.to(device), img_labels.to(device)
            aud_inputs, aud_labels = aud_data
            aud_inputs = aud_inputs.type(torch.FloatTensor)
            aud_inputs, aud_labels = aud_inputs.to(device), aud_labels.to(device)
            #print(aud_inputs.shape, img_inputs.shape)
            #torch.Size([24, 1, 40]) torch.Size([24, 3, 256, 512])
            if (torch.equal(img_labels, aud_labels) is False):
                print("----------------- ISSUE -----------------");
                # zero the parameter gradients
            optimizer.zero_grad()
            outputs = net(img_inputs, aud_inputs)
            loss = criterion(outputs, img_labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            t_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += img_labels.size(0)
            correct += (predicted == img_labels).sum().item()
            if i % 2 == 1:    # print every 2000 mini-batches
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / 2))
                running_loss = 0.0

        train_loss.append(t_loss)
        train_accuracy.append(100*correct/total)
        print("Accuracy of the network on train set is : %d %%" %(100 *correct/total))
        
    correct = 0
    total = 0
    running_loss = 0.0
    with torch.no_grad():
        for i, data in enumerate(loader['test'], 0):
            img_data, aud_data = data
            img_inputs, img_labels = img_data
            img_inputs, img_labels = img_inputs.to(device), img_labels.to(device)
            aud_inputs, aud_labels = aud_data
            aud_inputs = aud_inputs.type(torch.FloatTensor)
            aud_inputs, aud_labels = aud_inputs.to(device), aud_labels.to(device)
            outputs = net(img_inputs, aud_inputs)
            _, predicted = torch.max(outputs.data, 1)
            loss = criterion(outputs, img_labels)
            running_loss+=loss.item()
            total += img_labels.size(0)
            correct += (predicted == img_labels).sum().item()
            if i % 2 == 1:    # print every 2000 mini-batches
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / 2))
                running_loss = 0.0
    
    print('Finished Training')
    np.save('joint_model_cat_train_accuracy', train_accuracy)
    np.save('joint_model_cat_train_loss', train_loss)
    np.save('joint_model_cat_val_accuracy', val_accuracy)
    np.save('joint_model_cat_val_loss', val_loss)

def test(loader, criterion, net, device, optimizer):
    correct = 0 
    total = 0
    nb_classes = 8
    confusion_matrix = torch.zeros(nb_classes, nb_classes)
    with torch.no_grad():
        for i, data in enumerate(loader['test'], 0):
            img_data, aud_data = data
            img_inputs, img_labels = img_data
            img_inputs, img_labels = img_inputs.to(device), img_labels.to(device)
            aud_inputs, aud_labels = aud_data
            aud_inputs = aud_inputs.type(torch.FloatTensor)
            aud_inputs, aud_labels = aud_inputs.to(device), aud_labels.to(device)
            outputs = net(img_inputs, aud_inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += img_labels.size(0)
            correct += (predicted == img_labels).sum().item()
            for t, p in zip(img_labels.view(-1), predicted.view(-1)):
                confusion_matrix[t.long(), p.long()] += 1

    torch.save(net, 'joint_cat.pth')
    np.save('joint_model_cat_confusion_matrix', confusion_matrix)
    print('Accuracy of the network on the test images: %d %%' % (
            100 * correct / total))

def joint_cat(folder):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print (device)
    
    # Load the data that needs to be analyzed
    img_data_transform = {
        'vid_train' : tv.transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225])
                    ]),
        'vid_val' : tv.transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225])
                ]),
        'vid_test' : tv.transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225])
                ])    
    }

    img_dataset = ['vid_train', 'vid_test', 'vid_val']
    aud_dataset = ['aud_train', 'aud_test', 'aud_val']

    image_data = {}
    audio_data = {}

    for x in img_dataset:
        image_data[x] = tv.datasets.ImageFolder(folder + '/' + x, transform=img_data_transform[x])

    for x in aud_dataset:
        audio_data[x] = tv.datasets.DatasetFolder(folder + '/' + x, loader=npy_loader, extensions=('.npy'))
    
    datasets = ['train', 'test', 'val']
    loader = {}
    for x, y, z in zip(img_dataset, aud_dataset, datasets):
        ds = MyDataset(image_data[x], audio_data[y])
        loader[z] = torch.utils.data.DataLoader(ds, batch_size=24, shuffle=True, num_workers=0)
    
    net = Net()
    net.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
    
    train(loader, criterion, net, device, optimizer)
    test(loader, criterion, net, device, optimizer)