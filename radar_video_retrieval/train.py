import argparse
import os
import random
import numpy as np
import torch
import utils
from model import Baseline
from dataset import RVDataset
from torch.utils.data import DataLoader
from spcaug import get_transforms
from vidaug import get_preprocessing_pipelines
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='/home/spi/xuyinsong/data/radar_dataset')
    parser.add_argument('--method', type=str, default='baseline')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--in_domain', action='store_true')
    parser.add_argument('--epoch', type=int, default=60)
    parser.add_argument('--val_every', type=int, default=5)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--weight_decay', type=float, default=5e-5)
    parser.add_argument('--output_dir', type=str, default="result")
    parser.add_argument('--no_freq_mask', action='store_true')
    parser.add_argument('--no_time_mask', action='store_true')
    parser.add_argument('--no_freq_flip', action='store_true')
    parser.add_argument('--no_time_pad', action='store_true')
    parser.add_argument('--no_mix_up', action='store_true')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    print('Setup:')
    for k, v in sorted(vars(args).items()):
        print('\t{}: {}'.format(k, v))

    print('Setting up model', end=' ', flush=True)
    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    radar_train_trans, radar_val_trans = get_transforms(args)
    video_train_trans, video_val_trans = get_preprocessing_pipelines()


    if args.in_domain:
        train_dataset = RVDataset(args.data_dir, 'in_domain_split/train.txt',radar_train_trans, video_train_trans)
        val_dataset = RVDataset(args.data_dir, 'in_domain_split/val.txt',radar_val_trans, video_val_trans)
        test_dataset = RVDataset(args.data_dir, 'in_domain_split/test.txt',radar_val_trans, video_val_trans)
    else:
        print('OOD Mode')
        train_dataset = RVDataset(args.data_dir, 'out_of_domain_split/train.txt',radar_train_trans, video_train_trans)
        val_dataset = RVDataset(args.data_dir, 'out_of_domain_split/val.txt',radar_val_trans, video_val_trans)
        test_dataset = RVDataset(args.data_dir, 'out_of_domain_split/test.txt',radar_val_trans, video_val_trans)
    args.num_classes = train_dataset.N_CLASSES
    print('train set length:   {}'.format(len(train_dataset)))
    train_loader = DataLoader(dataset=train_dataset,batch_size=args.batch_size,shuffle=False,drop_last=False,num_workers=4,pin_memory=True)
    val_loader = DataLoader(dataset=val_dataset,batch_size=args.batch_size,shuffle=False,drop_last=False,num_workers=4,pin_memory=True)
    test_loader = DataLoader(dataset=test_dataset,batch_size=args.batch_size,shuffle=False,drop_last=False,num_workers=4,pin_memory=True)
    model = Baseline(train_loader, test_loader, test_loader, args)
    print('Ok')

    print('Training')
    model.train()

    print('Testing')
    model.test()

    train_stats = model.get_train_stats()
    
    test_loss = train_stats['loss']['test']
    r1 = train_stats['result']['R1']
    r5 = train_stats['result']['R5']
    r10 = train_stats['result']['R10']
    mr = train_stats['result']['MR']
    print(f'\ttest loss: {test_loss:.5f}, R1: {r1:.5f}, R5: {r5:.5f},, R10: {r10:.5f}, MR: {mr:.5f}')

    print('Saving training stats', end=' ', flush=True)
    utils.save_train_stats(train_stats, args.output_dir + '/train_stats.pkl')
    with open(args.output_dir + '/train_info.txt', 'w') as f:
        for k, v in sorted(vars(args).items()):
            f.write('{}: {}\n'.format(k, v))
        f.write('test_loss: {:.5f}\n'.format(test_loss))
        f.write('r1: {:.5f}\n'.format(r1))
        f.write('r5: {:.5f}\n'.format(r5))
        f.write('r10: {:.5f}\n'.format(r10))
        f.write('mr: {:.5f}\n'.format(mr))
    print('Ok')
