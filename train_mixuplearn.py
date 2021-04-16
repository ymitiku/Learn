from src.models import MixupModel
from src.utils import MixupDataset
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
    for (inputs1, inputs2), labels in dataloader:
        inputs1 = inputs1.to(device)
        inputs2 = inputs2.to(device)
        labels = labels.to(device).view(-1, 1)
        
        outputs = model([inputs1, inputs2])
        
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
        

def log_stats(path, epoch, train_loss, train_acc):
    file_exists = os.path.exists(path)
     

    with open(path, "a") as log_file:
        if not file_exists:
            log_file.write("epoch,train-loss,train-acc\n")
        log_file.write("{},{},{}\n".format(epoch, train_loss, train_acc))
        
def save_ckpt(args, model, optimizer, scheduler, epoch):

    ckpts_path = os.path.join(args.expdir, "checkpoints")
    if not os.path.exists(ckpts_path):
        os.mkdir(ckpts_path)
    if (epoch % 10) == 0:
        output_path = os.path.join(ckpts_path, "ckpt-{}.pth".format(epoch))
        torch.save({"model":model, "optimizer":optimizer, "scheduler":scheduler, "epoch":epoch}, output_path)
    output_path = os.path.join(args.expdir, "ckpt.pth")
    torch.save({"model":model, "optimizer":optimizer, "scheduler":scheduler, "epoch":epoch}, output_path)
    
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
    parser.add_argument("--logdir", type=str, default="./logs-mixlearn", help="Directory to save logging information")
    parser.add_argument("--expdir", type=str, default="./exps-mixlearn", help="Directory to save experiment information")
    parser.add_argument("--resume", type=str2bool, nargs='?',const=True, default=False,help="Resume from last  checkpoint")
    args = parser.parse_args()
    return args
def load_state_from_file(args):
    state = torch.load(os.path.join(args.expdir, "checkpoints", "ckpt.pth"))
    model = state["model"]
    optimizer = state["optimizer"]
    scheduler = state["scheduler"]
    start_epoch = state["epoch"] + 1
    
    return start_epoch, model, optimizer, scheduler
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
        transforms.RandomRotation(30),
        transforms.RandomGrayscale(),
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
    
    traindataset = MixupDataset(trainset)
    trainloader = DataLoader(traindataset, batch_size = args.batch_size, shuffle=True)
    if args.resume and os.path.exists(os.path.join(args.expdir, "checkpoints", "ckpt.pth")):
        start_epoch, model, optimizer, scheduler = load_state_from_file(args)
        print("Resuming from state: start-epoch: {}".format(start_epoch))
    else:
        model = MixupModel()
        model = model.to(device)
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)
        start_epoch = 0
    
    criterion = torch.nn.BCEWithLogitsLoss()

    
    for epoch in range(start_epoch, args.epochs):
        print("Epoch: {}/{}".format(epoch+1, args.epochs))
        train_loss, train_accuracy = train(model, trainloader, optimizer, criterion)
        scheduler.step()
        log_stats(log_path, epoch, train_loss, train_accuracy)
        save_ckpt(args,model,optimizer, scheduler, epoch)
        traindataset.build_mixup_dataset()
        
    
if __name__ == "__main__":
    main(); 