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
                output = self.model(x)
                loss = self.loss(output, y)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

        # self.model.cpu()

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
        #self.ALA.adaptive_local_aggregation(received_global_model, self.model)

    def set_parameters(self, model, progress):

        # Get class-specific prototypes from the local model
        local_prototypes = [[] for _ in range(self.num_classes)]
        batch_size = 16  # or any other suitable value
        trainloader = self.load_train_data(batch_size=batch_size)

        # print(f'client{id}')
        for x_batch, y_batch in trainloader:
            x_batch = x_batch.to(self.device)
            y_batch = y_batch.to(self.device)

            with torch.no_grad():
                proto_batch = self.model.base(x_batch)

            # Scatter the prototypes based on their labels
            for proto, y in zip(proto_batch, y_batch):
                local_prototypes[y.item()].append(proto)

        mean_prototypes = []

        # print(f'client{self.id}')
        for class_prototypes in local_prototypes:

            if not class_prototypes == []:
                # Stack the tensors for the current class
                stacked_protos = torch.stack(class_prototypes)

                # Compute the mean tensor for the current class
                mean_proto = torch.mean(stacked_protos, dim=0)
                mean_prototypes.append(mean_proto)
            else:
                mean_prototypes.append(None)

        # Align global model's prototype with the local prototype
        alignment_optimizer = torch.optim.SGD(model.base.parameters(), lr=0.01)  # Adjust learning rate and optimizer as needed
        alignment_loss_fn = torch.nn.MSELoss()

        # print(f'client{self.id}')
        for _ in range(1):  # Iterate for 1 epochs; adjust as needed
            for x_batch, y_batch in trainloader:
                x_batch = x_batch.to(self.device)
                y_batch = y_batch.to(self.device)
                global_proto_batch = model.base(x_batch)
                loss = 0
                for label in y_batch.unique():
                    if mean_prototypes[label.item()] is not None:
                        loss += alignment_loss_fn(global_proto_batch[y_batch == label], mean_prototypes[label.item()])
                alignment_optimizer.zero_grad()
                loss.backward()
                alignment_optimizer.step()

        # Substitute the parameters of the base, enabling personalization
        for new_param, old_param in zip(model.base.parameters(), self.model.base.parameters()):
            old_param.data = new_param.data.clone()


        # end

