from matplotlib.pyplot import axis
import torch
import time
from torchvision.models import resnet18
from backbone.cnn import Identity
from backbone.r18 import R18 
from mixup import mixup_criterion, mixup_data
from loss import MaxMarginRankingLoss
from metrics import compute_metrics, print_computed_metrics
import torch.nn.functional as F
import numpy as np

class RadarNet(torch.nn.Module):
    def __init__(self, args):
        super(RadarNet, self).__init__()
        self.backbone = resnet18(pretrained=True).to(args.device)
        self.backbone.fc = Identity()
    def forward(self, x):
        x = self.backbone(x)
        return F.normalize(x,dim=-1)

class VideoNet(torch.nn.Module):
    def __init__(self, args):
        super(VideoNet, self).__init__()
        self.backbone = R18().to(args.device)
    def forward(self, x):
        x = self.backbone(x)
        return F.normalize(x,dim=-1)


class Baseline:
    def __init__(self, train_loader, val_loader, test_loader, args):
        self.device = args.device
        self.args = args
        
        self.radar_net = RadarNet(args)
        self.video_net = VideoNet(args)

        self.loss_fn = MaxMarginRankingLoss(margin=0.1)

        parameters = list(self.radar_net.parameters()) + list(self.video_net.parameters())
        self.optimizer = torch.optim.SGD(parameters, lr=args.lr, weight_decay=0.0005 ,momentum=0.9) #torch.optim.Adam(self.network.parameters(),lr=args.lr,weight_decay=args.weight_decay)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=40)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.stats = {'loss': {'train': [], 'val': [], 'test': []},
                      'acc':  {'train': [], 'val': [], 'test': []},
                      'time': {'train': []},
                      'result':{'R1':[], 'R5':[],'R10':[],'MR':[]}}


    def train(self):
        val_every = 1 #self.args.val_every
        run_train_time, run_train_loss, run_train_acc = 0.0, 0.0, 0.0
        max_val_acc = -1

        for epoch in range(self.args.epoch):
            train_time = time.time()
            # Training
            
            for index, train_batch in enumerate(self.train_loader):
                # if not self.args.no_mix_up:
                #     train_loss, train_acc = self._train_step_mix(train_batch)
                # else:
                train_loss, train_acc = self._train_step_wo_mix(train_batch)
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
                    torch.save(self.radar_net.state_dict(), self.args.output_dir + '/best_radar_model.pt')
                    torch.save(self.video_net.state_dict(), self.args.output_dir + '/best_video_model.pt')
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

    #def _train_step(self, train_batch):
    def _train_step_wo_mix(self, train_batch):
        is_train = True
        self.radar_net.train(is_train)
        self.video_net.train(is_train)
        radar, video, y, _ = train_batch
        radar, video, y = radar.to(self.device), video.to(self.device), y.to(self.device)
        with torch.set_grad_enabled(is_train):
            feat_radar = self.radar_net(radar)
            feat_video = self.video_net(video)
            sim_matrix= feat_video@feat_radar.t()
            loss = self.loss_fn(sim_matrix)
            self.optimizer.zero_grad()
            loss.backward()
            train_loss = loss.item()
            train_acc = compute_metrics(sim_matrix.cpu().detach().numpy())['R1'] 
            self.optimizer.step()

        return train_loss, train_acc

    # def _train_step_mix(self, train_batch):
    #     is_train = True
    #     self.network.train(is_train)
    #     x, y, _ = train_batch
    #     with torch.set_grad_enabled(is_train):
    #         x, labels_a, labels_b, lam = mixup_data(x, y, 0.4)
    #         x, labels_a, labels_b = x.cuda(), labels_a.cuda(), labels_b.cuda()
    #         logits = self.network(self.backbone(x))
    #         loss_func = mixup_criterion(labels_a, labels_b, lam)
    #         loss = loss_func(self.loss_fn, logits)
    #         self.optimizer.zero_grad()
    #         loss.backward()
    #         predicted = torch.max(logits, 1)[1]
    #         train_loss = loss.item()
    #         train_acc = lam * predicted.eq(labels_a.view_as(predicted)).sum().item() + (1 - lam) * predicted.eq(labels_b.view_as(predicted)).sum().item()
    #         self.optimizer.step()

    #     return train_loss, train_acc


    def _validation_step(self):
        is_train = False
        val_loss, val_acc = 0.0, 0.0

        self.radar_net.train(is_train)
        self.video_net.train(is_train)
        r_feats, v_feats = [], []
        with torch.set_grad_enabled(is_train):
            for batch in self.val_loader:
                radar, video, y, _ = batch
                radar, video, y = radar.to(self.device), video.to(self.device), y.to(self.device)
                feat_radar = self.radar_net(radar)
                feat_video = self.video_net(video)
                r_feats.append(feat_radar.cpu().numpy())
                v_feats.append(feat_video.cpu().numpy())
                sim_matrix= feat_video@feat_radar.t()
                loss = self.loss_fn(sim_matrix)       
                val_loss += loss.item() * radar.size(0)

        r_feats = np.concatenate(r_feats, axis=0)
        v_feats = np.concatenate(v_feats, axis=0)
        m = np.matmul(r_feats, v_feats.T)
        val_acc = compute_metrics(m)['R1']  
        val_loss /= len(self.val_loader.dataset)
        return val_loss, val_acc

    def vis_results(self, names, results, path):
        f = open(path, 'w')
        for name, result in zip(names, results):
            f.write(name + ' ' + str(result) + '\r\n')
        f.close()

    def test(self):
        is_train = False
        test_loss, r1, r5, r10, mr = 0.0, 0.0, 0.0, 0.0 , 0.0

        self.radar_net.train(is_train)
        self.video_net.train(is_train)

        self.radar_net.load_state_dict(torch.load(self.args.output_dir + '/best_radar_model.pt'))
        self.video_net.load_state_dict(torch.load(self.args.output_dir + '/best_video_model.pt'))
        r_feats, v_feats = [], []
        with torch.set_grad_enabled(is_train):
            for batch in self.test_loader:

                radar, video, y, _ = batch
                radar, video, y = radar.to(self.device), video.to(self.device), y.to(self.device)
                feat_radar = self.radar_net(radar)
                feat_video = self.video_net(video)
                r_feats.append(feat_radar.cpu().numpy())
                v_feats.append(feat_video.cpu().numpy())
                sim_matrix= feat_video@feat_radar.t()
                loss = self.loss_fn(sim_matrix)
                test_loss += loss.item() * radar.size(0)
                # name.extend(list(path))
                # result.extend(list(predictions.cpu() == targets.cpu()))
        r_feats = np.concatenate(r_feats, axis=0)
        v_feats = np.concatenate(v_feats, axis=0)
        m = np.matmul(r_feats, v_feats.T)
        result = compute_metrics(m)
        #'result':{'R1':[], 'R5':[],'R10':[],'MR':[]}}
        #self.vis_results(name, result,self.args.output_dir + '/result.txt')
        self.stats['loss']['test'] = test_loss / len(self.test_loader.dataset)
        self.stats['result']['R1'] = result['R1']
        self.stats['result']['R5'] = result['R5']
        self.stats['result']['R10'] = result['R10']
        self.stats['result']['MR'] = result['MR']

    def get_train_stats(self):
        return self.stats
