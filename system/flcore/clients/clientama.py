import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time
import copy
from flcore.clients.clientbase import Client
from torch.autograd import grad


class clientAMA(Client):

    def __init__(self, args, id, train_samples, test_samples, **kwargs):
        super().__init__(args, id, train_samples, test_samples, **kwargs)

        self.global_model = self.model # (self.model.base, self.model.head)
        self.local_model = self.model.local_partitions

    def setGrad(self, mode):
        """mode in ('common', 'specific')"""
        t = {'common':True, 'specific':False}

        for p in self.model.parameters():
            p.requires_grad = t[mode]
        for part in self.local_model:
            for p in part.parameters():
                p.requires_grad = not t[mode]

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

                output = self.model.forward_both(x)
                loss1 = self.loss(output[0], y)
                loss2 = self.loss(output[1], y)

                self.optimizer.zero_grad()
                loss = loss1 + loss2
                loss.backward()
                self.optimizer.step()


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

