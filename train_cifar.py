from src.models import MixupLearnModel
import torch
from torchvision import transforms
from torchvision import datasets
from torch.utils.data import Dataset, DataLoader

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

def train(model, dataloader, optimizer, criterion, epoch):
    model.train()
    losses = 0
    corrects = 0
    total = 0
    for inputs, labels in dataloader:
        inputs = inputs.to(device)
        labels = labels.to(device)
        
        outputs = model(inputs)
        
        loss = criterion(outputs, labels)
        
        optimizer.zero_grads()
        loss.backward()
        optimizer.step()
        
        _, pred = torch.max(outputs, dim=1)
        size = labels.size(0)
        losses += loss.item() * size 
        corrects += (pred == labels).sum()
        total += size
    print("Epoch: {} loss: {:.4f} accuracy: {:.2f}%".format(epoch, losses/total, corrects * 100/total))
    
    return losses/total, corrects/total
        
def evaluate(model, dataloader,  criterion, epoch):
    model.eval()
    losses = 0
    corrects = 0
    total = 0
    for inputs, labels in dataloader:
        inputs = inputs.to(device)
        labels = labels.to(device)
        
        outputs = model(inputs)
        
        loss = criterion(outputs, labels)
        
        
        _, pred = torch.max(outputs, dim=1)
        size = labels.size(0)
        losses += loss.item() * size 
        corrects += (pred == labels).sum()
        total += size
    print("Epoch: {} loss: {:.4f} accuracy: {:.2f}%".format(epoch, losses/total, corrects * 100/total))
    
        
        
        
def main():
    model = MixupLearnModel(10)
    model = model.to(device)
    cifar_normalize = transforms.Normalize(
        (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))

    transform_cifar_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        cifar_normalize
    ])

    transform_cifar_test = transforms.Compose([
        transforms.ToTensor(),
        cifar_normalize
    ])

    trainset = datasets.CIFAR10(root='./data',
                                train=True,
                                download=True,
                                transform=transform_cifar_train)
    testset = datasets.CIFAR10(root='./data',
                                train=False,
                                download=True,
                                transform=transform_cifar_test)
    trainloader = DataLoader(trainset, batch_size = 128, shuffle=True)
    testloader = DataLoader(trainset, batch_size = 128, shuffle=True)
    
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3, momentum=0.9)
    criterion = torch.nn.CrossEntropyLoss()
    
    for i in range(100):
        train_loss, train_accuracy = train(model, trainloader, optimizer, criterion, i+1)
        val_loss, val_accuracy = train(model, testloader,  criterion, i+1)
    
if __name__ == "__main__":
    main(); 