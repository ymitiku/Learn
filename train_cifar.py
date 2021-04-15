from src.models import ClassificationModel
import torch
from torchvision import transforms
from torchvision import datasets
from torch.utils.data import Dataset, DataLoader
import argparse
import numpy as np 
import os
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

def train(model, dataloader, optimizer, criterion):
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
    print("Training: loss: {:.4f} accuracy: {:.2f}%({}/{})".format(losses/total, corrects * 100/total, corrects, total))
    
    return losses/total, corrects/total
        
def evaluate(model, dataloader,  criterion):
    model.eval()
    losses = 0
    corrects = 0
    total = 0
    with torch.no_grad():
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
    print("Validation: loss: {:.4f} accuracy: {:.2f}%({}/{})".format(losses/total, corrects * 100/total, corrects, total))
    return losses/total, corrects/total
def log_stats(path, epoch, train_loss, train_acc, test_loss, test_acc):
    file_exists = os.path.exists(path)
     

    with open(path, "a") as log_file:
        if not file_exists:
            log_file.write("epoch,train-loss,train-acc,test-loss,test-acc\n")
        log_file.write("{},{},{},{},{}\n".format(epoch, train_loss, train_acc, test_loss, test_acc))
        
def save_ckpt(args, model, optimizer, scheduler, epoch, best_acc):

    output_folder = os.path.join(args.expdir, "checkpoints")
    if not os.path.exists(output_folder):
        os.mkdir(output_folder)
    output_path = os.path.join(output_folder, "ckpt.pth")
    torch.save({"model":model, "optimizer":optimizer, "scheduler":scheduler, "epoch":epoch, "best_acc":best_acc}, output_path)
def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() == "true":
        return True
    elif v.lower() == "false":
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=200, help="Number of epoch to train the model ")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=0.1, help="Initial learning rate")
    parser.add_argument("--momentum", type=float, default=0.9, help="Initial learning rate")
    parser.add_argument("--weight_decay", type=float, default=5e-4, help="Initial learning rate")
    parser.add_argument("--logdir", type=str, default="./logs", help="Directory to save logging information")
    parser.add_argument("--expdir", type=str, default="./exps", help="Directory to save experiment information")
    parser.add_argument("--resume", type=str2bool, nargs='?',const=True, default=False,help="Resume from last best checkpoint")
    args = parser.parse_args()
    return args
def load_state_from_file(args):
    state = torch.load(os.path.join(args.expdir, "checkpoints", "ckpt.pth"))
    best_acc = state["best_acc"]
    model = state["model"]
    optimizer = state["optimizer"]
    scheduler = state["scheduler"]
    start_epoch = state["epoch"] + 1
    
    return start_epoch, model, optimizer, scheduler, best_acc
def main():
    
    args = get_args()
    
    if not os.path.exists(args.logdir):
        os.mkdir(args.logdir)
    if not os.path.exists(args.expdir):
        os.mkdir(args.expdir)
    log_path = os.path.join(args.logdir, "train.log")
    if not args.resume and os.path.exists(log_path):
        os.remove(log_path)
        
    
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
    if args.resume and os.path.exists(os.path.join(args.expdir, "checkpoints", "ckpt.pth")):
        start_epoch, model, optimizer, scheduler, best_acc = load_state_from_file(args)
        print("Resuming from state: start-epoch: {} best-acc: {:.2f}%".format(start_epoch, best_acc % 100))
    else:
        model = ClassificationModel(10)
        model = model.to(device)
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)
        best_acc = 0
        start_epoch = 0
    
    criterion = torch.nn.CrossEntropyLoss()

    
    for epoch in range(start_epoch, args.epochs):
        print("Epoch: {}/{}".format(epoch+1, args.epochs))
        train_loss, train_accuracy = train(model, trainloader, optimizer, criterion)
        val_loss, val_accuracy = evaluate(model, testloader,  criterion)
        scheduler.step()
        log_stats(log_path, epoch, train_loss, train_accuracy, val_loss, val_accuracy)
        if val_accuracy > best_acc:
            print("Found best accuracy:{:.2f} saving to disk".format(best_acc * 100))
            best_acc = val_accuracy
            save_ckpt(args,model, optimizer, scheduler, epoch, best_acc)
        
    
if __name__ == "__main__":
    main(); 