import os
import cv2
import matplotlib.pyplot as plt
import xmltodict
import random
from os import listdir
from os.path import isfile, join
import torchvision
import torch
import torchvision.models as models
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch.optim as optim
def createDirectory(dirname):
    try:
        os.mkdir(dirname)
    except FileExistsError:
        print("Directory " + dirname + " already exists.")

def training():
    models_dir = "models/"
    createDirectory(models_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device("cpu")
    device


    model = models.resnet50(pretrained=True)

    for layer, param in model.named_parameters():

        if 'layer4' not in layer:
            param.requires_grad = False

    model.fc = torch.nn.Sequential(torch.nn.Linear(2048, 512),
                                   torch.nn.ReLU(),
                                   torch.nn.Dropout(0.2),
                                   torch.nn.Linear(512, 5),
                                   torch.nn.LogSoftmax(dim=1))

    train_transforms = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))
        ])

    dir_name = "./train/"
    dataset = datasets.ImageFolder(dir_name, transform = train_transforms)

    val_dir= "./validation/"
    val_data = datasets.ImageFolder(val_dir, transform = train_transforms)

    BATCH_SIZE = 20
    train_loader = torch.utils.data.DataLoader(dataset,
                                              batch_size=BATCH_SIZE,
                                              shuffle=True)
    validation_loader = torch.utils.data.DataLoader(val_data,
                                                   batch_size=BATCH_SIZE,
                                                   shuffle=True)

    learning_r = 0.01
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr = learning_r)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

    model.to(device)

    total_epoch = 30

    best_epoch = 0
    training_losses = []
    val_losses = []
    accuracy_list = []

    for epoch in range(total_epoch):

        epoch_train_loss = 0
        scheduler.step()
        for X, y in train_loader:
            X, y = X.cpu(), y.cpu()

            optimizer.zero_grad()
            result = model(X)

            loss = criterion(result, y)
            epoch_train_loss += loss.item()
            loss.backward()
            optimizer.step()

        training_losses.append(epoch_train_loss)

        epoch_val_loss = 0
        correct = 0
        total = 0

        with torch.no_grad():
            for X, y in validation_loader:
                X, y = X.cpu(), y.cpu()
                result = model(X)
                loss = criterion(result, y)
                epoch_val_loss += loss.item()
                _, maximum = torch.max(result.data, 1)
                total += y.size(0)
                correct += (maximum == y).sum().item()

        val_losses.append(epoch_val_loss)
        accuracy = correct / total
        accuracy_list.append(accuracy)
        print("EPOCH:", epoch, ", Training Loss:", epoch_train_loss, ", Validation Loss:", epoch_val_loss, ", Accuracy: ",
              accuracy)

        if min(val_losses) == val_losses[-1]:
            best_epoch = epoch
            checkpoint = {'model': model,
                          'state_dict': model.state_dict(),
                          'optimizer': optimizer.state_dict()}

            torch.save(checkpoint, models_dir + '{}.pth'.format(epoch))
            print("Model saved")


def load_checkpoint(filepath):
    checkpoint = torch.load(filepath)
    model = checkpoint['model']
    model.load_state_dict(checkpoint['state_dict'])
    for parameter in model.parameters():
        parameter.requires_grad = False

    return model.eval()

