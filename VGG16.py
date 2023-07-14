"""An implementation of a vision transformer for breast cancer classification

The original code have been taken from:

https://blog.paperspace.com/vgg-from-scratch-pytorch/
"""


from torchvision import transforms

# Define the data augmentation transform
transform = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])



#import numpy as np
import torch
import torch.nn as nn
from torchvision import datasets
from torchvision import transforms
#from torch.utils.data.sampler import SubsetRandomSampler

# Defining the path to the dataset
train_root = '/home/aimsgh-02/Music/split_data/with_val/train'
test_root = '/home/aimsgh-02/Music/split_data/with_val/test'
valid_root = '/home/aimsgh-02/Music/split_data/with_val/valid'


# Loading the dataset
train_dataset = datasets.ImageFolder(
    root=train_root,
    transform=transform
)

test_dataset = datasets.ImageFolder(
    root=test_root,
    transform=transform
)
valid_dataset = datasets.ImageFolder(
    root=valid_root,
    transform=transform
)

# Specifying batch size and creating data loaders
batch_size = 10
train_loader = torch.utils.data.DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True
)

test_loader = torch.utils.data.DataLoader(
    test_dataset,
    batch_size=batch_size,
    shuffle=False
)

valid_loader = torch.utils.data.DataLoader(
    valid_dataset,
    batch_size=batch_size,
    shuffle=False
)

class VGG16(nn.Module):
    def __init__(self, num_classes=2):
        super(VGG16, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU())
        self.layer2 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(), 
            nn.MaxPool2d(kernel_size = 2, stride = 2))
        self.layer3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU())
        self.layer4 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 2, stride = 2))
        self.layer5 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU())
        self.layer6 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU())
        self.layer7 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 2, stride = 2))
        self.layer8 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU())
        self.layer9 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU())
        self.layer10 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 2, stride = 2))
        self.layer11 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU())
        self.layer12 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU())
        self.layer13 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 2, stride = 2))
        self.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(7*7*512, 4096),
            nn.ReLU())
        self.fc1 = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            nn.ReLU())
        self.fc2= nn.Sequential(
            nn.Linear(4096, num_classes))
        
    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.layer5(out)
        out = self.layer6(out)
        out = self.layer7(out)
        out = self.layer8(out)
        out = self.layer9(out)
        out = self.layer10(out)
        out = self.layer11(out)
        out = self.layer12(out)
        out = self.layer13(out)
        out = out.reshape(out.size(0), -1)
        out = self.fc(out)
        out = self.fc1(out)
        out = self.fc2(out)
        return out

num_classes = 2
num_epochs = 10
#batch_size = 100
learning_rate = 0.005


# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = VGG16(num_classes).to(device)


# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, weight_decay = 0.005, momentum = 0.9)  


# Train the model
total_step = len(train_loader)
print('The total number of steps is: ',total_step)

import time



#Training the model

total_step = len(train_loader)
step = 0

# Start the timer
start_time = time.time()

for epoch in range(num_epochs):
    
    train_loss_values = []
    train_accuracy_values = []
    train_loss, train_correct, train_total = 0, 0, 0
    for i, (images, labels) in enumerate(train_loader):  
        # Move tensors to the configured device
        images = images.to(device)
        labels = labels.to(device)
        

        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        
        
        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        step += 1
        if step % 100 == 0:
            print('Step', step)
            print('hello world')
                   
        train_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        train_total += labels.size(0)
        train_correct += (predicted == labels).sum().item()
        
        
    # Calculate the average loss per epoch
    epoch_loss = train_loss / len(train_loader)
    # Calculate the accuracy per epoch
    epoch_accuracy = 100*train_correct / train_total
        
    train_loss_values.append(epoch_loss)  # Append the average loss per epoch to the list


    train_accuracy_values.append(epoch_accuracy)  # Append the accuracy per epoch to the list

    print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}' 
                   .format(epoch+1, num_epochs, i+1, total_step, loss.item()))
    
    # Validation
    with torch.no_grad():
        correct = 0
        total = 0
        valid_loss = 0
        val_loss_values = []
        val_accuracy_values = []
        TP, TN, FP, FN = 0, 0, 0, 0
        for images, labels in valid_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            valid_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            FP += ((predicted == 1) & (labels == 0)).sum().item() 
            FN += ((predicted == 0) & (labels == 1)).sum().item() 
            TP += ((predicted == 1) & (labels == 1)).sum().item() 
            TN += ((predicted == 0) & (labels == 0)).sum().item() 
            del images, labels, outputs
            
        valid_accuracy = 100 * correct / total
        valid_loss /= len(valid_loader)
        val_loss_values.append(valid_loss)
        val_accuracy_values.append(valid_accuracy)
        print('Accuracy of the network on the {} validation images: {} %'.format(len(valid_loader), 100 * correct / total)) 
        print('Precision of the network on the {} validation images: {} %'.format(len(valid_loader), 100 * TP / (TP+FP))) 
        print('Recall of the network on the {} validation images: {} %'.format(len(valid_loader), 100 * TP / (TP+FN))) 
        print('False positive rate of the network on the {} validation images: {} %'.format(len(valid_loader), 100 * FP / (FP+TN)))


end_time = time.time()  
    
# Calculate the running time
running_time = end_time - start_time

# Print or log the running time
print(f"Running time: {running_time:.2f} seconds")

with torch.no_grad():
    correct = 0
    total = 0
    TP, TN, FP, FN = 0, 0, 0, 0 
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        FP += ((predicted == 1) & (labels == 0)).sum().item() 
        FN += ((predicted == 0) & (labels == 1)).sum().item() 
        TP += ((predicted == 1) & (labels == 1)).sum().item() 
        TN += ((predicted == 0) & (labels == 0)).sum().item() 
        del images, labels, outputs

    print('Accuracy of the network on the {} test images: {} %'.format(len(test_loader), 100 * correct / total))  
    print('Precision of the network on the {} test images: {} %'.format(len(valid_loader), 100 * TP / (TP+FP))) 
    print('Recall of the network on the {} test images: {} %'.format(len(valid_loader), 100 * TP / (TP+FN))) 
    print('False positive rate of the network on the {} test images: {} %'.format(len(valid_loader), 100 * FP / (FP+TN))) 
    
 
 
import matplotlib.pyplot as plt



# Plot the loss
plt.plot(range(1, num_epochs + 1), train_loss_values, label='Train')
plt.plot(range(1, num_epochs + 1), val_loss_values, label='Validation')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Loss Evolution')
plt.legend()
plt.show()

# Plot the accuracy
plt.plot(range(1, num_epochs + 1), train_accuracy_values, label='Train')
plt.plot(range(1, num_epochs + 1), val_accuracy_values, label='Validation')
plt.xlabel('Epochs')
plt.ylabel('Accuracy (%)')
plt.title('Accuracy Evolution')
plt.legend()
plt.show()
   
    
    
