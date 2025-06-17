import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time
import copy
from flcore.clients.clientbase import Client
from torch.autograd import grad

class FedAMALocalCNN(nn.Module):
    def __init__(self, in_features=1, num_classes=10, dim=1024, p=0.5):
        super().__init__()

        self.fc1 = nn.Sequential(
            nn.Linear(dim, 512*p),
            nn.ReLU(inplace=True)
        )
        self.fc = nn.Linear(512*(1 + p), num_classes)

    def forward(self, x):
        out = self.fc1(out)
        out = self.fc(out)
        return out

def constructLocalModel(model):
    pass


class clientAMA(Client):

    def __init__(self, args, id, train_samples, test_samples, **kwargs):
        super().__init__(args, id, train_samples, test_samples, **kwargs)

        self.global_model = self.model # (self.model.base, self.model.head)
        self.local_model = constructLocalModel(self.model) # (self.model.base, local_head)

        self.global_optimizer = torch.optim.SGD(self.global_model.parameters(),
                                                lr=self.learning_rate)
        self.local_optimizer = torch.optim.SGD(self.local_model.parameters(),
                                                lr=self.learning_rate)

    def train(self):
        trainloader = self.load_train_data()
        # self.model.to(self.device)
        self.model.train()
        
        start_time = time.time()

        max_local_epochs = self.local_epochs
        if self.train_slow:
            max_local_epochs = np.random.randint(1, max_local_epochs // 2)

        for epoch in range(max_local_epochs):
            for i, (x, y) in enumerate(trainloader):
                if type(x) == type([]):
                    x[0] = x[0].to(self.device)
                else:
                    x = x.to(self.device)
                y = y.to(self.device)
                if self.train_slow:
                    time.sleep(0.1 * np.abs(np.random.rand()))

                # Sequential

                for p in global_model.parameters():
                    p.requires_grad = True
                for p in local_model.parameters():
                    p.requires_grad = False

                output = self.global_model(x)
                loss = self.loss(output, y)
                self.global_optimizer.zero_grad()
                loss.backward()
                self.global_optimizer.step()


                for p in global_model.parameters():
                    p.requires_grad = False
                for p in local_model.parameters():
                    p.requires_grad = True

                output = self.local_model(x)
                loss = self.loss(output, y)
                self.local_optimizer.zero_grad()
                loss.backward()
                self.local_optimizer.step()


        if self.learning_rate_decay:
            self.learning_rate_scheduler.step()

        self.train_time_cost['num_rounds'] += 1
        self.train_time_cost['total_cost'] += time.time() - start_time

    def evaluate(self):
        testloader = self.load_test_data()
        self.model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for x, y in testloader:
                x = x.to(self.device)
                y = y.to(self.device)
                outputs = self.model(x)
                _, predicted = outputs.max(1)
                total += y.size(0)
                correct += predicted.eq(y).sum().item()
        accuracy = 100. * correct / total
        return accuracy

    # def set_parameters(self, model, progress):
        # # Substitute the parameters of the base, enabling personalization
        # for new_param, old_param in zip(model.base.parameters(), self.model.base.parameters()):
        #     old_param.data = new_param.data.clone()


    def local_initialization(self, received_global_model):
        pass

    def set_parameters(self, model, progress):
        return

        # Substitute the parameters of the base, enabling personalization
        for new_param, old_param in zip(model.base.parameters(), self.model.base.parameters()):
            old_param.data = new_param.data.clone()

