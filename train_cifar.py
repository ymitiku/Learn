from src.models import ClassificationModel
import torch
from torchvision import transforms
from torchvision import datasets
from torch.utils.data import Dataset, DataLoader
import argparse
import numpy as np 
import os
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
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        _, pred = torch.max(outputs, dim=1)
        size = labels.size(0)
        losses += loss.item() * size 
        corrects += (pred == labels).sum()
        total += size
    print("Training: {} loss: {:.4f} accuracy: {:.2f}%".format(epoch, losses/total, corrects * 100/total))
    
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
    print("Validation: {} loss: {:.4f} accuracy: {:.2f}%".format(epoch, losses/total, corrects * 100/total))
    return losses/total, corrects/total
def log_stats(path, train_loss, train_acc, test_loss, test_acc):
    file_exists = os.path.exists(path)
     

    with open(path, "a+") as log_file:
        if not file_exists:
            log_file.write("train-loss,train-acc,test-loss,test-acc\n")
        log_file.write("{},{},{},{}\n".format(train_loss, train_acc, test_loss, test_acc))
        
def save_ckpt(args, model, optimizer, scheduler, epoch, best_acc):
    output_folder = os.path.join(args.expdir, "checkpoints")
    output_path = os.path.join(output_folder, "ckpt.pth")
    torch.save({"model":model, "optimizer":optimizer, "scheduler":scheduler, "epoch":epoch, "best_acc":best_acc}, output_path)
def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=200, help="Number of epoch to train the model ")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3, help="Initial learning rate")
    parser.add_argument("--logdir", type=str, default="./logs", help="Directory to save logging information")
    parser.add_argument("--expdir", type=str, default="./exps", help="Directory to save experiment information")
        
    args = parser.parse_args()

def main():
    
    args = get_args()
    
    if not os.path.exists(args.logdir):
        os.mkdir(args.logdir)
    if not os.path.exists(args.expdir):
        os.mkdir(args.expdir)
    log_path = os.path.join(args.logdir, "train.log")
    
    model = ClassificationModel(10)
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
    trainloader = DataLoader(trainset, batch_size = args.batch_size, shuffle=True)
    testloader = DataLoader(testset, batch_size = args.batch_size, shuffle=False)
    
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)

    criterion = torch.nn.CrossEntropyLoss()
    epochs = 100
    best_acc = 0
    for epoch in range(epochs):
        print("Epoch: {}/{}".format(i+1, epochs))
        train_loss, train_accuracy = train(model, trainloader, optimizer, criterion, epoch+1)
        val_loss, val_accuracy = evaluate(model, testloader,  criterion, epoch+1)
        scheduler.step()
        log_stats(log_path, train_loss, train_accuracy, val_loss, val_accuracy)
        if val_accuracy > best_acc:
            save_ckpt(args,model, optimizer, scheduler, epoch, best_acc)
            best_acc = val_accuracy
        
    
if __name__ == "__main__":
    main(); 