import os
import random
import numpy as np
import argparse
from sklearn.model_selection import train_test_split


def save_split(xs, ys, path):
    f = open(path, 'w')
    for x, y in zip(xs, ys):
        f.write(x + ' ' + str(y) + '\r\n')
    f.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='/home/spi/xuyinsong/lip')
    parser.add_argument('--val_size', type=float, default=0.1)
    parser.add_argument('--test_size', type=float, default=0.2)
    parser.add_argument('--output_dir', type=str, default='radar_classification_merge/in_domain_split')#
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    random.seed(args.seed)
    np.random.seed(args.seed)

    data, labels = [], []
    ids= ['datahjn','datapjb','dataxys']#['0','1','2','3','4','5','6','7','7','9']
    for id in ids:
        with open(os.path.join(args.data_dir, id+'.txt')) as f:
            for line in f:
                path, label = line.strip().split(' ')
                data.append(os.path.join(id+'_doppler_npy',path + '.npy'))
                labels.append(int(label))
    # X_train, X_test, y_train, y_test
    train, test, y_train, y_test = train_test_split(data, labels, test_size=args.test_size, stratify=labels)
    train, val, y_train, y_val = train_test_split(train, y_train, test_size=args.val_size/(1-args.test_size), stratify=y_train)
    
    save_split(train, y_train, f'{args.output_dir}/train.txt')
    save_split(val, y_val, f'{args.output_dir}/val.txt')
    save_split(test, y_test, f'{args.output_dir}/test.txt')

    # import pdb
    # pdb.set_trace()

    # np.savetxt(f'{args.output_dir}/train.txt', sorted(train), fmt='%s')
    # np.savetxt(f'{args.output_dir}/val.txt', sorted(val), fmt='%s')
    # np.savetxt(f'{args.output_dir}/test.txt', sorted(test), fmt='%s')

