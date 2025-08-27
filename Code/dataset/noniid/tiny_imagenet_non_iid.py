import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from dataset.tiny_imagenet import TinyImageNet
from dataset.utils import shuffle_list


def get_tiny_imagenet(data_dir):
    data_train = TinyImageNet(data_dir, train=True)
    data_test = TinyImageNet(data_dir, train=False)

    x_train, y_train = np.array(data_train.imgs), np.array(data_train.labels)
    x_test, y_test = np.array(data_test.imgs), np.array(data_test.labels)

    return x_train, y_train, x_test, y_test


def print_image_data_stats_train(data_train, labels_train):
    print("\nData: ")
    print(" - Train Set: ({},{}), Labels: {},..,{}".format(
        len(data_train), len(labels_train), np.min(labels_train), np.max(labels_train)))


def print_image_data_stats_test(data_test, labels_test):
    print(" - Test Set: ({},{}), Labels: {},..,{}".format(
        len(data_test), len(labels_test), np.min(labels_test), np.max(labels_test)))


def split_image_data_dirichlet(data, labels, n_clients=100, dirichlet_level=0.1, verbose=True):
    n_classes = labels.max() + 1

    alpha = dirichlet_level
    label_distribution = np.random.dirichlet([alpha] * n_clients, n_classes)

    class_indices = [np.where(labels == i)[0] for i in range(n_classes)]

    client_indices = [[] for _ in range(n_clients)]
    for c, fracs in zip(class_indices, label_distribution):
        for i, idx in enumerate(np.split(c, (np.cumsum(fracs)[:-1] * len(c)).astype(int))):
            client_indices[i] += [idx]

    client_indices = [np.concatenate(idx) for idx in client_indices]

    clients_split = []
    for indc in client_indices:
        clients_split.append([data[indc], labels[indc]])

    sample_number = np.zeros(n_clients, dtype=int)

    def print_split(clients_split):
        print("Data split:")
        for i, client in enumerate(clients_split):
            client_labels = client[1]

            split = np.zeros(n_classes, dtype=int)

            for label in range(n_classes):
                split[label] = np.sum(client_labels == label)
            sample_number[i] = np.sum(split)
            print(" - Client {}: {}".format(i, split.tolist()))
        print(f" - Total : {sample_number.tolist()}")

    print_split(clients_split)

    return clients_split, sample_number.tolist()


class CustomImageDataset(Dataset):

    def __init__(self, inputs, labels, transforms=None):
        assert inputs.shape[0] == labels.shape[0]
        self.inputs = inputs
        self.labels = labels.tolist()
        self.transforms = transforms

    def __getitem__(self, index):
        img, label = self.inputs[index], self.labels[index]

        if self.transforms is not None:
            img = self.transforms(img)

        return img, label

    def __len__(self):
        return self.inputs.shape[0]


def get_data_loaders_train_tinet(data_dir, nclients, batch_sizes, classes_pc=200, verbose=True, transforms_train=None,
                                 dirichlet_level=0.1, transforms_eval=None, noniid=None):
    x_train, y_train, _, _ = get_tiny_imagenet(data_dir)

    if verbose:
        print_image_data_stats_train(x_train, y_train)

    split = None
    sample_number = None
    print('Non diid is ' + str(noniid))
    if noniid == 'dirichlet':
        split, sample_number = split_image_data_dirichlet(x_train, y_train, n_clients=nclients,
                                                          dirichlet_level=dirichlet_level,
                                                          verbose=verbose)
    else:
        raise NotImplementedError('Non-iid type not implemented.')

    split_tmp = shuffle_list(split)

    client_loaders = [torch.utils.data.DataLoader(CustomImageDataset(x, y, transforms_train),
                                                  batch_size=bs, shuffle=True, drop_last=True) for (x, y), bs in zip(
        split_tmp, batch_sizes)]

    return client_loaders, sample_number


def get_data_loaders_test_tinet(data_dir, nclients, batch_sizes, classes_pc=200, verbose=True, transforms_train=None,
                                dirichlet_level=0.1, transforms_eval=None, noniid=None):
    _, _, x_test, y_test = get_tiny_imagenet(data_dir)
    if verbose:
        print_image_data_stats_test(x_test, y_test)

    test_loader = torch.utils.data.DataLoader(CustomImageDataset(x_test, y_test, transforms_eval),
                                              batch_size=batch_sizes, shuffle=False, )

    return test_loader
