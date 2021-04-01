import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torch.optim.lr_scheduler import _LRScheduler
from torch.utils.data import DataLoader, sampler, random_split

import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models

import numpy as np
import random
import time

from attribute_extractor import *

import sys

SEED = 42
start_ids = [1, 10, 25, 40, 55, 59, 74, 80, 95, 106, 121, 136, 150, 153, 168, 183, 198, 213, 218, 223, 237, 241, 245,
             249, 264, 279, 294, 309]
end_ids = [9, 24, 39, 54, 58, 73, 79, 94, 105, 120, 135, 149, 152, 167, 182, 197, 212, 217, 222, 236, 240, 244, 248,
           263, 278, 293, 308, 312]

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True

trainval_set = open('ZSL/trainval_fname.txt').readlines()
trainval_set = [fname[:-1] for fname in trainval_set]
test_seen = open('ZSL/test_seen.txt').readlines()
test_seen = [fname[:-1] for fname in test_seen]
test_unseen = open('ZSL/test_unseen.txt').readlines()
test_unseen = [fname[:-1] for fname in test_unseen]

with open('ZSL/new_class_attribute.txt', 'r') as f:
    lines = f.readlines()
mtx = [list(map(float, l[:-1].split(' '))) for l in lines]
class_attribute = torch.tensor(mtx) / 100

transform = transforms.Compose([transforms.Resize(255),
                                transforms.CenterCrop(224),
                                transforms.ToTensor()])


pretrained_model = models.resnet101(pretrained=True)
pretrained_model.fc = nn.Identity()

print(f'The model has {count_parameters(model):,} trainable parameters')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
batch_size = 64
criterion = torch.nn.MultiLabelSoftMarginLoss()

for s, e in zip(start_ids, end_ids):
    feature_range = [s - 1, e]
    model_path = 'model_%s_%s.pt' % (s, e)

    trainval_data = datasets.ImageFolder("CUB_200_2011/images/", transform=transform,
                                         is_valid_file=lambda x: x in trainval_set,
                                         target_transform=lambda x: class2attribute(x, feature_range))
    test_seen_data = datasets.ImageFolder("CUB_200_2011/images/", transform=transform,
                                          is_valid_file=lambda x: x in test_seen,
                                          target_transform=lambda x: class2attribute(x, feature_range))
    test_unseen_data = datasets.ImageFolder("CUB_200_2011/images/", transform=transform,
                                            is_valid_file=lambda x: x in test_unseen,
                                            target_transform=lambda x: class2attribute(x, feature_range))
    train_data_len = int(len(trainval_data) * 0.75)
    valid_data_len = int((len(trainval_data) - train_data_len))
    train_data, val_data = random_split(trainval_data, [train_data_len, valid_data_len])

    model = AttributeNet(pretrained_model, e - s + 1, 64)
    model = model.to(device)
    criterion = criterion.to(device)

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=True)
    test_seen_loader = DataLoader(test_seen_data, batch_size=batch_size, shuffle=True)
    test_unseen_loader = DataLoader(test_unseen_data, batch_size=batch_size, shuffle=True)
    classes = trainval_data.classes

    model = AttributeNet(pretrained_model, e - s + 1, 64)
    model = model.to(device)
    criterion = criterion.to(device)

    FOUND_LR = 1e-3
    params = [
        {'params': model.pretrained.conv1.parameters(), 'lr': FOUND_LR / 10},
        {'params': model.pretrained.bn1.parameters(), 'lr': FOUND_LR / 10},
        {'params': model.pretrained.layer1.parameters(), 'lr': FOUND_LR / 8},
        {'params': model.pretrained.layer2.parameters(), 'lr': FOUND_LR / 6},
        {'params': model.pretrained.layer3.parameters(), 'lr': FOUND_LR / 4},
        {'params': model.pretrained.layer4.parameters(), 'lr': FOUND_LR / 2},
        {'params': model.fc.parameters()},
        {'params': model.final_fc.parameters()},
    ]

    criterion = torch.nn.MultiLabelSoftMarginLoss()
    optimizer = optim.Adam(params, lr=FOUND_LR)

    EPOCHS = 10
    STEPS_PER_EPOCH = len(train_loader)
    TOTAL_STEPS = EPOCHS * STEPS_PER_EPOCH

    MAX_LRS = [p['lr'] for p in optimizer.param_groups]

    scheduler = lr_scheduler.OneCycleLR(optimizer,
                                        max_lr=MAX_LRS,
                                        total_steps=TOTAL_STEPS)

    test_seen_loss, test_seen_acc = evaluate(model, test_seen_loader, criterion, device)
    test_unseen_loss, test_unseen_acc = evaluate(model, test_unseen_loader, criterion, device)

    print(f'Test seen Loss: {test_seen_loss:.3f} | Test seen Acc @1: {test_seen_acc * 100:.3f}')
    print(f'Test unseen Loss: {test_unseen_loss:.3f} | Test unseen Acc @1: {test_unseen_acc * 100:.3f}')

    best_valid_loss = float('inf')

    for epoch in range(EPOCHS):
        print(epoch)

        start_time = time.time()

        train_loss, train_acc = train(model, train_loader, optimizer, criterion, scheduler, device)
        valid_loss, valid_acc = evaluate(model, val_loader, criterion, device)

        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            torch.save(model.state_dict(), model_path)

        end_time = time.time()

        epoch_mins, epoch_secs = epoch_time(start_time, end_time)

        print(f'Epoch: {epoch + 1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s')
        print(f'\tTrain Loss: {train_loss:.3f} | Train Acc: {train_acc * 100:.3f} ')
        print(f'\tValid Loss: {valid_loss:.3f} | Valid Acc: {valid_acc * 100:.3f} ')

    model.load_state_dict(torch.load(model_path))

    test_seen_loss, test_seen_acc = evaluate(model, test_seen_loader, criterion, device)
    test_unseen_loss, test_unseen_acc = evaluate(model, test_unseen_loader, criterion, device)

    print(f'Test seen Loss: {test_seen_loss:.3f} | Test seen Acc @1: {test_seen_acc * 100:.3f}')
    print(f'Test unseen Loss: {test_unseen_loss:.3f} | Test unseen Acc @1: {test_unseen_acc * 100:.3f}')

    del model
    del criterion
    del trainval_data
    del test_seen_data
    del test_unseen_data
    torch.cuda.empty_cache()