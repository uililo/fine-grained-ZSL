import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models

import numpy as np
import random

from attribute_extractor import *


SEED = 42

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True

start_ids = [1,10,25,40,55,59,74,80, 95,106,121,136,150,153,168,183,198,213,218,223,237,241,245,249,264,279,294,309]
end_ids =  [ 9,24,39,54,58,73,79,94,105,120,135,149,152,167,182,197,212,217,222,236,240,244,248,263,278,293,308,312]

pretrained_model = models.resnet101(pretrained = True)
pretrained_model.fc = nn.Identity()

transform = transforms.Compose([transforms.Resize(255),
                                transforms.CenterCrop(224),
                                transforms.ToTensor()])

all_data = datasets.ImageFolder("CUB_200_2011/images/", transform=transform)
all_loader = DataLoader(all_data, batch_size=64, shuffle=False)


for i in range(28):
    s = start_ids[i]
    e = end_ids[i]
    model = AttributeNet(pretrained_model, attribute_dim=e-s+1, embedding_dim=64)
    model_path = 'model_%d_%d.pt'%(s,e)
    model.load_state_dict(torch.load(model_path))

    device = 'cuda'
    model = model.to('cuda')
    model.eval()
    print('start forwarding with %s' % model_path)

    attr_prob = []
    with torch.no_grad():
        for i, (x, y) in enumerate(all_loader):
            if i % 10 == 0:
                print(i)
            x = x.to(device)
            y_pred = model.forward(x)
            attr_prob.append(y_pred.cpu().numpy())
    np.savetxt('attribute_probabilities/attribute_%d_%d.txt' % (s, e), np.concatenate(attr_prob, axis=0))

    model.final_fc = nn.Identity()
    attr_emb = []
    with torch.no_grad():
        for i, (x, y) in enumerate(all_loader):
            if i%10 == 0:
                print(i)
            x = x.to(device)
            y_pred = model.forward(x)
            attr_emb.append(y_pred.cpu().numpy())
    np.savetxt('attribute_embeddings/attribute_%d_%d.txt'%(s,e),np.concatenate(attr_emb,axis=0))

