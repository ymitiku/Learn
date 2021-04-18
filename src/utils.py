from torch.utils.data import Dataset, DataLoader
import torch 
import numpy as np 
device = torch.device("cuda") if torch.cuda.is_available() else torch.device('cpu')

def vec_mul_ten(vec, tensor):
    size = list(tensor.size())
    size[0] = -1
    size_rs = [1 for i in range(len(size))]
    size_rs[0] = -1
    vec = vec.reshape(size_rs).expand(size)
    res = vec * tensor
    return res

def mixup_data(x, k = 4):
    '''Returns mixed inputs, lists of targets, and lambdas'''
    batch_size = x.size(0)
    
    first_indices = torch.randperm(batch_size)
    second_half = first_indices[batch_size:]
    second_indices = first_indices.clone()
    second_indices[batch_size:] = second_half[torch.randperm(len(second_half))]
    
    
    
    labels = torch.ones((first_indices.size(0),))
    labels[batch_size:] = 1.0
    
    first_indices = first_indices.to(device)
    second_indices = second_indices.to(device)
    x = x.to(device)
    lams_dist = torch.distributions.Dirichlet(torch.tensor([1/k] * k))
    
    lams1 = lams_dist.sample([len(x)])
    lams2 = lams_dist.sample([len(x)])
    
    lams1 = lams1.to(device)
    lams2 = lams2.to(device)

    mixed_x_1 = vec_mul_ten(lams1[:, 0], x[first_indices])
    mixed_x_2 = vec_mul_ten(lams2[:, 0], x[second_indices])
    
    for i in range(1, k):
        
        batch_size = x.size()[0]
        index1 = torch.randperm(batch_size).to(device)
        index2 =  torch.randperm(batch_size).to(device)
        
        
        mixed_x_1 += vec_mul_ten(lams1[:, i], x[index1])
        mixed_x_2 += vec_mul_ten(lams2[:, i], x[index2])
        
        
        
    return mixed_x_1.cpu(), mixed_x_2.cpu(), labels.cpu()


class MixupDataset(Dataset):
    def __init__(self, dataset, transform = None):
        super().__init__()
        self.dataset = dataset
        self.transform = transform
        self.build_mixup_dataset()
    def build_mixup_dataset(self):
        dataloader = DataLoader(self.dataset, batch_size = len(self.dataset), shuffle=True)
        assert len(dataloader) == 1
        for images, _ in dataloader:
            print("Mixing up dataset")
            self.mixed_images1, self.mixed_images2, self.labels = mixup_data(images, k=2)
            print("Done!")
        
    def __len__(self):
        return len(self.dataset)
    def __getitem__(self, index):
        image1 = self.mixed_images1[index]
        image2 = self.mixed_images2[index]
        label = self.labels[index]
        
        if self.transform is not None:
            image1 = self.transform(image1)
            image2 = self.transform(image2)
        return (image1, image2), label
        