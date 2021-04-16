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
    
    rand_indices = torch.stack([torch.randperm(batch_size) for i in range(k)], dim=0).cuda()
    
    rand_bits = torch.from_numpy(np.random.choice([0, 1], p=[0.82, 0.18], size=rand_indices.shape)).cuda()
    
    rand_indices_2 = rand_indices.clone()
    
    rand_indices_2[rand_bits==0] = torch.randint(0, batch_size, ((rand_bits==0).sum().item(),))
    
    labels = torch.all(rand_bits == 0, dim=0)
    
    
    x = x.to(device)
    lams_dist = torch.distributions.Dirichlet([1/k] * k)
    
    lams1 = lams_dist.sample([len(x)])
    lams2 = lams_dist.sample([len(x)])
    
    lams1 = lams1.to(device)
    lams2 = lams2.to(device)

    mixed_x_1 = None
    mixed_x_2 = None
    

    for i in range(0, k):
        batch_size = x.size()[0]
        index = torch.randperm(batch_size).to(device)
        index1 = rand_indices[i]
        index2 = rand_indices_2[i]
        if mixed_x_1 is None:
            mixed_x_1 = vec_mul_ten(lams1[i, :], x[index1])
            mixed_x_2 = vec_mul_ten(lams2[i, :], x[index2])
        else:
            mixed_x_1 += vec_mul_ten(lams1[i, :], x[index1])
            mixed_x_2 += vec_mul_ten(lams2[i, :], x[index2])
            
        
        
    return mixed_x_1, mixed_x_2


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
            self.mixed_images1, self.mixed_images2, self.labels = mixup_data(images, k=4)
        
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
        