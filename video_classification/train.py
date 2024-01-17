import argparse
import os
import random
import numpy as np
import torch
import utils
from model import Baseline
from dataset import VideoDataset
from torch.utils.data import DataLoader
from video_aug import get_preprocessing_pipelines
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='/home/spi/xuyinsong/radar_dataset')
    parser.add_argument('--method', type=str, default='baseline')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--in_domain', action='store_true')
    parser.add_argument('--epoch', type=int, default=60)
    parser.add_argument('--val_every', type=int, default=5)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--weight_decay', type=float, default=5e-5)
    parser.add_argument('--output_dir', type=str, default="result")
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

    train_trans, val_trans = get_preprocessing_pipelines()
    if args.in_domain:
        train_dataset = VideoDataset(args.data_dir, 'in_domain_split/train.txt',train_trans)
        val_dataset = VideoDataset(args.data_dir, 'in_domain_split/val.txt',val_trans)
        test_dataset = VideoDataset(args.data_dir, 'in_domain_split/test.txt',val_trans)
    else:
        print('OOD Mode')
        train_dataset = VideoDataset(args.data_dir, 'out_of_domain_split/train.txt',train_trans)
        val_dataset = VideoDataset(args.data_dir, 'out_of_domain_split/val.txt',val_trans)
        test_dataset = VideoDataset(args.data_dir, 'out_of_domain_split/test.txt',val_trans)
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
    test_acc, test_loss = train_stats['acc']['test'], train_stats['loss']['test']
    print(f'\ttest loss: {test_loss:.5f}, test acc: {test_acc:.5f}')

    print('Saving training stats', end=' ', flush=True)
    utils.save_train_stats(train_stats, args.output_dir + '/train_stats.pkl')
    with open(args.output_dir + '/train_info.txt', 'w') as f:
        for k, v in sorted(vars(args).items()):
            f.write('{}: {}\n'.format(k, v))
        f.write('test_loss: {:.5f}\n'.format(test_loss))
        f.write('test_acc: {:.5f}\n'.format(test_acc))
    print('Ok')
