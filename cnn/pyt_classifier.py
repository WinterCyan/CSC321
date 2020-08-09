import torch
import torchvision
import torch.utils.data
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np

def imshow(img):
    img = img/2+0.5
    npimg = img.numpy() # shape: 3x36x138
    plt.imshow(np.transpose(npimg,(1,2,0))) # shape: 36x138x3, RGB img
    plt.show()

transform = transforms.Compose(
    [transforms.ToTensor(),transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))]
)
trainset = torchvision.datasets.CIFAR10(root='./data',train=True,download=True,transform=transform)
trainloader = torch.utils.data.DataLoader(trainset,batch_size=4,shuffle=True,num_workers=2)
testset = torchvision.datasets.CIFAR10(root='./data',train=False,download=True,transform=transform)
testloader = torch.utils.data.DataLoader(testset,batch_size=4,shuffle=False,num_workers=2)
classes = ('plane','car','bird','cat','deer','dog','frog','horse','ship','truck')

# dataiter = iter(trainloader)
# images, labels = dataiter.next() # load a batch
# print(' '.join('%5s' % classes[labels[j]] for j in range(4)))
# imshow(torchvision.utils.make_grid(images))

class Net(nn.Module):
    # define structure
    def __init__(self):
        super(Net,self).__init__()
        self.conv1 = nn.Conv2d(3,6,5)
        self.pool = nn.MaxPool2d(2,2)
        self.conv2 = nn.Conv2d(6,16,5)
        self.fc1 = nn.Linear(16*5*5,120)
        self.fc2 = nn.Linear(120,84)
        self.fc3 = nn.Linear(84,10)

    # define forward pass
    def forward(self,x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1,16*5*5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# train pass
net = Net()
criterion = nn.CrossEntropyLoss() # loss function
optimizer = optim.SGD(net.parameters(),lr = 0.001,momentum=0.9) # optimizer
for epoch in range(2):
    running_loss = 0.0
    for i, data in enumerate(trainloader,0): # load batches
        inputs, labels = data
        # print(inputs.shape) # 4x3x32x32, four imgs
        optimizer.zero_grad() # clear up weights
        outputs = net(inputs) # forward pass
        loss = criterion(outputs,labels) # calculate loss
        loss.backward() # backward pass
        optimizer.step() # update weights
        running_loss += loss.item()
        if i % 2000 == 1999:
            print('[%d,%5d] loss: %.3f'%(epoch+1,i+1,running_loss/2000))
            running_loss = 0.0

print('Finished.')
path = '../cifar_net.pth'
torch.save(net.state_dict(),path) # save model

# test pass
net = Net()
net.load_state_dict(torch.load(path)) # load model
correct = 0
total = 0
with torch.no_grad():
    for data in testloader:
        images,labels = data
        outputs = net(images) # predict, outputs: 4x10, 10 labels for 4 imgs
        _,predicted = torch.max(outputs.data,1) # predicted: 4, max-value's index for every img
        total+=labels.size(0)
        # predicted==labels: [False, True, True, True],
        # .sum(): tensor(3)
        # .item(): 3
        correct+=(predicted==labels).sum().item()
print('Accuracy of the network on the 10000 test images: %d %%' % (100 * correct / total))