import torch
import time
from torchvision.models import resnet18
from backbone.cnn import  Identity
from backbone.mstcn import MSTCN
from backbone.r18 import R18
class Baseline:
    def __init__(self, train_loader, val_loader, test_loader, args):
        self.device = args.device
        self.args = args
        #self.network = resnet18(pretrained=True).to(args.device)
        self.backbone = R18().to(args.device)
        # self.backbone.requires_grad_(False)
        #self.backbone.eval()
        self.network = torch.nn.Linear(512, args.num_classes).to(args.device)
        self.loss_fn = torch.nn.CrossEntropyLoss()
        parameters = list(self.network.parameters()) + list(self.backbone.parameters())
        self.optimizer = torch.optim.SGD(parameters, lr=args.lr, weight_decay=0.0005 ,momentum=0.9) #torch.optim.Adam(self.network.parameters(),lr=args.lr,weight_decay=args.weight_decay)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=40)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.stats = {'loss': {'train': [], 'val': [], 'test': []},
                      'acc':  {'train': [], 'val': [], 'test': []},
                      'time': {'train': []}}


    def train(self):
        val_every = 1 #self.args.val_every
        run_train_time, run_train_loss, run_train_acc = 0.0, 0.0, 0.0
        max_val_acc = -1

        for epoch in range(self.args.epoch):
            train_time = time.time()
            # Training
            
            for index, train_batch in enumerate(self.train_loader):
                train_loss, train_acc = self._train_step(train_batch)
            self.scheduler.step()
            run_train_time += (time.time() - train_time)
            run_train_loss += train_loss
            run_train_acc += train_acc

            if (epoch+1) % val_every == 0 or epoch == (self.args.epoch-1):
                # Validation
                val_loss, val_acc = self._validation_step()
                # Save model when the validation accuracy increases
                if val_acc > max_val_acc:
                    max_val_acc = val_acc
                    torch.save(self.network.state_dict(), self.args.output_dir + '/best_model.pt')
                if (epoch+1) % val_every == 0:
                    run_train_time /= val_every
                    run_train_loss /= val_every
                    run_train_acc /= val_every
                elif epoch == (self.args.epoch-1):
                    n_steps = epoch
                    n_steps -= n_steps//val_every * val_every
                    run_train_time /= n_steps
                    run_train_loss /= n_steps
                    run_train_acc /= n_steps

                self.stats['loss']['train'].append(run_train_loss)
                self.stats['acc']['train'].append(run_train_acc)
                self.stats['loss']['val'].append(val_loss)
                self.stats['acc']['val'].append(val_acc)
                self.stats['time']['train'].append(run_train_time)

                # Print stats
                print(f'\tepoch {epoch+1:>5}/{self.args.epoch}: '
                      f'train loss: {run_train_loss:.5f}, '
                      f'train acc: {run_train_acc:.5f} | '
                      f'val loss: {val_loss:.5f}, '
                      f'val acc: {val_acc:.5f} | '
                      f'iter time: {run_train_time:.5f}')

                run_train_time, run_train_loss, run_train_acc = 0.0, 0.0, 0.0

    def _train_step(self, train_batch):
        is_train = True
        self.network.train(is_train)
        x, y, _ = train_batch
        x, y = x.to(self.device), y.to(self.device)
        with torch.set_grad_enabled(is_train):
            outputs = self.network(self.backbone(x))
            loss = self.loss_fn(outputs, y)
            self.optimizer.zero_grad()
            loss.backward()
            predictions = torch.max(outputs, 1)[1]
            train_loss = loss.item()
            train_acc = (predictions == y).float().mean().item()
            self.optimizer.step()

        return train_loss, train_acc

    def _validation_step(self):
        is_train = False
        val_loss, val_acc = 0.0, 0.0

        self.network.train(is_train)
        
        with torch.set_grad_enabled(is_train):
            for batch in self.val_loader:
                x, y, path = batch
                x, y = x.to(self.device), y.to(self.device)
                outputs = self.network(self.backbone(x))
                loss = self.loss_fn(outputs, y)

                predictions = torch.max(outputs, 1)[1]
                val_loss += loss.item() * x.size(0)
                val_acc += (predictions == y).sum().item()
                
        val_loss /= len(self.val_loader.dataset)
        val_acc /= len(self.val_loader.dataset)

        return val_loss, val_acc

    def vis_results(self, names, results, path):
        f = open(path, 'w')
        for name, result in zip(names, results):
            f.write(name + ' ' + str(result) + '\r\n')
        f.close()

    def test(self):
        is_train = False
        test_loss, test_acc = 0.0, 0.0

        self.network.train(is_train)
        self.network.load_state_dict(torch.load(self.args.output_dir + '/best_model.pt'))
        name, result = [], [] 
        with torch.set_grad_enabled(is_train):
            for batch in self.test_loader:
                inputs, targets, path = batch
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                outputs = self.network(self.backbone(inputs))
                loss = self.loss_fn(outputs, targets)

                predictions = torch.max(outputs, 1)[1]
                test_loss += loss.item() * inputs.size(0)
                test_acc += (predictions == targets).sum().item()
                # name.extend(list(path))
                # result.extend(list(predictions.cpu() == targets.cpu()))

        #self.vis_results(name, result,self.args.output_dir + '/result.txt')
        self.stats['loss']['test'] = test_loss / len(self.test_loader.dataset)
        self.stats['acc']['test'] = test_acc / len(self.test_loader.dataset)

    def get_train_stats(self):
        return self.stats
