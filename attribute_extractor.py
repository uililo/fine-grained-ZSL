import torch.nn as nn
import torch


class AttributeNet(nn.Module):
    def __init__(self, pretrained_model, attribute_dim, embedding_dim):
        super().__init__()
        self.pretrained = pretrained_model
        self.emb_dim = embedding_dim
        self.attr_dim = attribute_dim

        self.fc = nn.Linear(2048, self.emb_dim)
        self.final_fc = nn.Linear(self.emb_dim, self.attr_dim)

    def forward(self, x):
        x = self.pretrained(x)
        x = self.fc(x)
        x = self.final_fc(x)

        return x


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def calculate_top1_acc(y_pred, y):
    with torch.no_grad():
        output = torch.nn.Softmax(dim=1)
        prob_pred = output(y_pred)
        return ((y.max(axis=1).indices) == (prob_pred.max(axis=1).indices)).float().sum().item() / len(y)


def train(model, iterator, optimizer, criterion, scheduler, device):
    epoch_loss = 0
    epoch_acc = 0

    model.train()

    for (x, y) in iterator:
        x = x.to(device)
        y = y.to(device)

        optimizer.zero_grad()

        y_pred = model(x)

        loss = criterion(y_pred, y)
        acc = calculate_top1_acc(y_pred, y)

        loss.backward()

        optimizer.step()

        scheduler.step()

        epoch_loss += loss.item()
        epoch_acc += acc

    epoch_loss /= len(iterator)
    epoch_acc /= len(iterator)

    return epoch_loss, epoch_acc


def evaluate(model, iterator, criterion, device):
    epoch_loss = 0
    epoch_acc = 0

    model.eval()

    with torch.no_grad():
        for (x, y) in iterator:
            x = x.to(device)
            y = y.to(device)

            y_pred = model(x)

            loss = criterion(y_pred, y)
            acc = calculate_top1_acc(y_pred, y)

            epoch_loss += loss.item()
            epoch_acc += acc

    epoch_loss /= len(iterator)
    epoch_acc /= len(iterator)

    return epoch_loss, epoch_acc


def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs