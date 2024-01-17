import hashlib
import pickle
import numpy as np
import torch
import matplotlib.pyplot as plt

def save_train_stats(stats, file_path):
    with open(file_path, 'wb') as f:
        pickle.dump(stats, f)


def load_train_stats(file_path):
    with open(file_path, 'rb') as f:
        stats = pickle.load(f)
    return stats


def plot_train_stats(stats, path):
    plt.figure(figsize=(9.6, 4.8))

    plt.subplot(1, 2, 1)
    plt.plot(stats['loss']['train'], 'b', label='Training')
    plt.xlabel('Epoch')
    plt.ylabel('Average Loss')
    plt.title('Training loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(stats['acc']['val'], 'r', label='Validation')
    plt.ylim([0.0, 100.0])
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.title('Validation accuracy')
    plt.legend()

    plt.savefig(path)
    plt.close()


def get_batch_class(dataset, class_label, batch_size, replace=True):
    class_indices = np.where(np.asarray(dataset.labels) == class_label)[0]
    batch_indices = np.random.choice(class_indices, batch_size, replace=replace)
    inputs = torch.stack([dataset[i][0] for i in batch_indices])
    targets = torch.tensor([dataset[i][1] for i in batch_indices], device=inputs.device)
    return inputs, targets


def seed_generator(*args):
    m = hashlib.md5(str(args).encode('utf-8'))
    h = m.hexdigest()
    i = int(h, 16)
    seed = i % (2**31)
    return seed
