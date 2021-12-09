from __future__ import print_function
from __future__ import division
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torchvision
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, models, transforms
#import matplotlib.pyplot as plt
import time
import os
import copy
import random

img_dir = "Your directory"
exp_folder = sorted(os.listdir(img_dir))
model_name = "vgg"
num_classes = 24
num_epochs = 60
feature_extract = False

random.seed(0)
np.random.seed(0)
torch.manual_seed(0)
if torch.cuda.is_available():
    torch.backends.cudnn.deterministic = True
    torch.cuda.manual_seed(0)
class Image_Dataset(Dataset):
    """AU image dataset."""
    def __init__(self, data=[], transform = None):
        'Initialization'
        #self.split = split
        self.data = []
        self.transform = transform
    def get_train(self):
        for identity in range(1, 26, 2):
        #for identity in range(1,(len(exp_folder)-1),2):
            id_folder = sorted(os.listdir(img_dir + exp_folder[identity]))
            for video in range(1,len(id_folder)-2):
                video_folder = sorted(os.listdir(img_dir + exp_folder[identity] + '/' +  id_folder[video]))
                for frame in range(0,len(video_folder),2):
                    image_x = img_dir + exp_folder[identity] + '/' + id_folder[video] +'/' + video_folder[frame +1]
                    output_i = img_dir + exp_folder[identity] + '/' + id_folder[video] +'/' + video_folder[frame]
                    self.data.append((image_x, output_i))


    def get_test(self):
        for identity in range(1,len(exp_folder)-1,2):
            id_folder = sorted(os.listdir(img_dir + exp_folder[identity]))
            for video in range(len(id_folder)-2, len(id_folder)):
                video_folder = sorted(os.listdir(img_dir + exp_folder[identity] + '/' + id_folder[video]))
                for frame in range(0,len(video_folder),2):
                    image_x = img_dir + exp_folder[identity] + '/' + id_folder[video] +'/' + video_folder[frame +1]
                    output_i = img_dir + exp_folder[identity] + '/' + id_folder[video] +'/' + video_folder[frame]
                    self.data.append((image_x, output_i))


    def __len__(self):
        'Denotes the total number of samples'
        return len(self.data)

    def __getitem__(self, idx):
        'Generates one sample of data'
        image, label  = self.data[idx]
        if self.transform:
           temp = Image.open(image)
           print('temp is', temp.size)
           img = self.transform(Image.open(image))
           print('size is', img.size())
        else:
           img = io.imread(image)
        output_i_raw = open(label)
        output_i = np.genfromtxt(output_i_raw)
        target = torch.tensor(output_i).type(torch.FloatTensor)
        #sample = {'image': image, 'params': params}
        return img, target

def train_model(model, dataloaders, criterion, optimizer, num_epochs):
    since = time.time()

    #val_acc_history = []

    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = 100000000

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
                if (epoch+1)%40==0:
                    for idx, param in enumerate(optimizer.param_groups):
                        param['lr'] = param['lr'] * 0.1
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0
            #running_corrects = 0
            total = 0
            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    # Get model outputs and calculate loss
                    # Special case for inception because in training it has an auxiliary output. In train
                    #   mode we calculate the loss by summing the final output and the auxiliary output
                    #   but in testing we only consider the final output.

                    outputs = model(inputs)
                    loss = criterion(outputs, labels)

                    #_, preds = torch.max(outputs, 1)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += (((labels-outputs)**2).sum()).item()
                total += labels.shape[0]

            epoch_loss = running_loss / total

            print('{} Loss: {:.4f}'.format(phase, epoch_loss))

            # deep copy the model
            if phase == 'val' and epoch_loss < best_loss:
                best_loss = epoch_loss
                best_model_wts = copy.deepcopy(model.state_dict())
                torch.save(model.state_dict(), "Your directory")
            #if phase == 'val':
                #val_acc_history.append(epoch_acc)

        #print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_loss))

    # load best model weights
#    model.load_state_dict(best_model_wts)
#    return model, val_acc_history

def test_model(model, dataloaders):
    test_loss = 0
    active_loss = 0
    non_active_loss = 0
    total_data = 0
    correct = 0
    correct_2 = 0
    active_sum = 0
    non_active_sum = 0
    total = 0
    total_2 = 0
    model.load_state_dict(torch.load("Your directory"))
    for inputs, labels in dataloaders['val']:
        inputs = inputs.to(device)
        labels = labels.to(device)
        with torch.set_grad_enabled(False):
            outputs = model(inputs)
            guesses = outputs.cpu().detach()
            answers = labels.cpu().detach()
            #print("prediction is " + str(guesses.size()))
            #print("answer is " + str(answers.size()))
            results = np.copy(abs(answers-guesses))
            pos_answers = np.copy(answers)
            pos_answers[pos_answers>0] = 1
            total += pos_answers[pos_answers>0].sum()
            pos_right = pos_answers*results
            correct += (np.logical_and(pos_right>0, pos_right<=.1).sum()).item()

            total_2 += answers.shape[0]*answers.shape[1]
            correct_2 += ((abs(answers-guesses)<= 0.1).sum()).item()

            test_loss += (((answers-guesses)**2).sum()).item()
            target = answers.reshape(-1,1)
            pred = guesses.reshape(-1, 1)
            unweighted_error = (target-pred)**2
            if len(torch.nonzero(target)) == 0:
                active_loss += 0
            else:
                active_loss += torch.sum(unweighted_error[torch.nonzero(target).long()[:,0]]).item()
            active_sum += len(torch.nonzero(target))
            non_active_sum += len((torch.nonzero(target==0)))
            non_active_loss += torch.sum(unweighted_error[torch.nonzero(target==0).long()[:,0]]).item()
            total_data += labels.shape[0]

    total_mse = test_loss/total_data
    active_mse = active_loss/total_data
    accuracy = 100*(correct/total)
    accuracy_2 = 100*(correct_2/total_2)
    non_active_mse = non_active_loss/total_data

    print('testing mse is ' + str(total_mse))
    print('active error is', (active_mse))
    print('non active error is', (non_active_mse))
    print('Active Accuracy is', accuracy)
    print('Total Accuracy is', accuracy_2)
    print('au MSE is', (test_loss/(active_sum + non_active_sum)))
    print('au active example  mse is', (active_loss/active_sum))
    print('au non active example mse is', (non_active_loss/non_active_sum))



def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False


def initialize_model(model_name, num_classes, feature_extract, use_pretrained=False):
    if model_name == "vgg":
        model_ft = models.vgg16_bn(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        #num_ftrs = model_ft.classifier[6].in_features
        model_ft.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 1000, bias=True),
            nn.ReLU(True),
            nn.Dropout(p=0.4),
            nn.Linear(1000, 512),
            nn.ReLU(True),
            nn.Dropout(p=0.3),
            nn.Linear(512, num_classes),
        )
        input_size = 448
    return model_ft, input_size


model_ft, input_size = initialize_model(model_name, num_classes, feature_extract, use_pretrained=True)


print("Initializing Datasets and Dataloaders...")


train = Image_Dataset(transform=transforms.Compose([
                                               transforms.RandomResizedCrop(448, scale=(0.8, 1.0)),
                                               transforms.ToTensor(),
                                               transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]))
train.get_train()
test = Image_Dataset(transform=transforms.Compose([
                                                transforms.CenterCrop(240),
                                                transforms.ToTensor(),
                                                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                                                ]))
test.get_test()
train_data = DataLoader(train, batch_size=1,shuffle=True, num_workers=0,pin_memory=True)
test_data = DataLoader(test, batch_size=1,shuffle=False, num_workers=0,pin_memory=True)
dataloaders_dict = {'train': train_data, 'val': test_data}
# Detect if we have a GPU available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model_ft = model_ft.to(device)

params_to_update = model_ft.parameters()
print("Params to learn:")
if feature_extract:
    params_to_update = []
    for name,param in model_ft.named_parameters():
        if param.requires_grad == True:
            params_to_update.append(param)
            print("\t",name)
else:
    for name,param in model_ft.named_parameters():
        if param.requires_grad == True:
            print("\t",name)

# Observe that all parameters are being optimized
optimizer_ft = optim.SGD([{'params': model_ft.features.parameters(), 'lr': 0.001, 'momentum': 0.9, 'weight_decay': 1e-4},
                          {'params': model_ft.classifier.parameters(), 'lr': 0.01, 'momentum': 0.9, 'weight_decay': 1e-4}])
criterion = nn.MSELoss()

# Train and evaluate
train_model(model_ft, dataloaders_dict, criterion, optimizer_ft, num_epochs=num_epochs)
test_model(model_ft, dataloaders_dict)


