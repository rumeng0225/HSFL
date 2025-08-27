import random

import numpy as np
import torch
import torchvision
from torch.utils.data import DataLoader, Dataset

from dataset.utils import generate_random_sample_proportions, shuffle_list


def get_cifar10(data_dir):
    data_train = torchvision.datasets.CIFAR10(data_dir, train=True, download=True)
    data_test = torchvision.datasets.CIFAR10(data_dir, train=False, download=True)

    x_train, y_train = data_train.data, np.array(data_train.targets)
    x_test, y_test = data_test.data, np.array(data_test.targets)

    return x_train, y_train, x_test, y_test


def print_image_data_stats_train(data_train, labels_train):
    print("\nData: ")
    print(" - Train Set: ({},{}), Range: [{:.3f}, {:.3f}], Labels: {},..,{}".format(
        data_train.shape, labels_train.shape, np.min(data_train), np.max(data_train),
        np.min(labels_train), np.max(labels_train)))


def print_image_data_stats_test(data_test, labels_test):
    print(" - Test Set: ({},{}), Range: [{:.3f}, {:.3f}], Labels: {},..,{}".format(
        data_test.shape, labels_test.shape, np.min(data_test), np.max(data_test),
        np.min(labels_test), np.max(labels_test)))


def clients_rand(train_len, nclients):
    client_tmp = []
    sum_ = 0

    for i in range(nclients - 1):
        tmp = random.randint(10, 100)
        sum_ += tmp
        client_tmp.append(tmp)

    client_tmp = np.array(client_tmp)

    clients_dist = ((client_tmp / sum_) * train_len).astype(int)
    num = train_len - clients_dist.sum()
    to_ret = list(clients_dist)
    to_ret.append(num)
    return to_ret


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


def split_image_data_quantity(data, labels, n_clients=100, sample_proportions=None, verbose=True):
    n_classes = labels.max() + 1

    class_indices = [np.where(labels == i)[0] for i in range(n_classes)]

    class_counts = [len(indices) for indices in class_indices]

    client_indices = [[] for _ in range(n_clients)]

    for class_id in range(n_classes):
        total_samples = class_counts[class_id]
        start_idx = 0

        for client_id, proportion in enumerate(sample_proportions):
            end_idx = start_idx + int(proportion * total_samples)

            if end_idx > total_samples:
                raise ValueError("Invalid sample proportions")

            client_indices[client_id].extend(class_indices[class_id][start_idx:end_idx])
            start_idx = end_idx

    clients_split = [[data[indices], labels[indices]] for indices in client_indices]

    sample_number = np.zeros(n_clients, dtype=int)

    def print_split(clients_split):
        print("Data split:")
        for i, client in enumerate(clients_split):
            client_labels = client[1]

            split = np.zeros(n_classes, dtype=int)

            for label in range(n_classes):
                split[label] = np.sum(client_labels == label)
            sample_number[i] = np.sum(split)
            print(" - Client {}: {}".format(i, split))

    print_split(clients_split)

    return clients_split, sample_number.tolist()


def split_image_data_realwd(data, labels, n_clients=100, verbose=True):
    def break_into(n, m):

        to_ret = [1 for i in range(m)]
        for i in range(n - m):
            ind = random.randint(0, m - 1)
            to_ret[ind] += 1
        return to_ret

    n_classes = len(set(labels))
    classes = list(range(n_classes))
    np.random.shuffle(classes)

    label_indcs = [list(np.where(labels == class_)[0]) for class_ in classes]
    n_data = data.shape[0]
    n_labels = np.max(labels) + 1

    tmp = [np.random.randint(1, 10) for _ in range(n_clients)]
    total_partition = sum(tmp)

    class_partition = break_into(total_partition, len(classes))

    class_partition = sorted(class_partition, reverse=True)
    class_partition_split = {}

    for ind, class_ in enumerate(classes):
        class_partition_split[class_] = [list(i) for i in np.array_split(label_indcs[ind], class_partition[ind])]

    clients_split = []
    count = 0
    for i in range(n_clients):
        n = tmp[i]
        j = 0
        indcs = []

        while n > 0:
            class_ = classes[j]
            if len(class_partition_split[class_]) > 0:
                indcs.extend(class_partition_split[class_][-1])
                count += len(class_partition_split[class_][-1])
                class_partition_split[class_].pop()
                n -= 1
            j += 1

        classes = sorted(classes, key=lambda x: len(class_partition_split[x]), reverse=True)
        if n > 0:
            raise ValueError(" Unable to fulfill the criteria ")
        clients_split.append([data[indcs], labels[indcs]])

    # print(class_partition_split)
    print("total example ", count)

    print(clients_split)

    sample_number = np.zeros(n_clients, dtype=int)

    def print_split(clients_split):
        print("Data split:")
        for i, client in enumerate(clients_split):
            split = np.sum(client[1].reshape(1, -1) == np.arange(n_labels).reshape(-1, 1), axis=1)
            sample_number[i] = np.sum(split)
            print(" - Client {}: {}".format(i, split))
        print()

    print_split(clients_split)
    return clients_split, sample_number.tolist()


def split_image_data(data, labels, n_clients=100, classes_per_client=10, shuffle=True, verbose=True):
    n_data = data.shape[0]
    n_labels = np.max(labels) + 1

    data_per_client = clients_rand(len(data), n_clients)
    data_per_client_per_class = [np.maximum(1, nd // classes_per_client) for nd in data_per_client]

    data_idcs = [[] for i in range(n_labels)]
    for j, label in enumerate(labels):
        data_idcs[label] += [j]
    if shuffle:
        for idcs in data_idcs:
            np.random.shuffle(idcs)

    clients_split = []
    c = 0
    for i in range(n_clients):
        client_idcs = []

        budget = data_per_client[i]
        c = np.random.randint(n_labels)
        while budget > 0:
            take = min(data_per_client_per_class[i], len(data_idcs[c]), budget)

            client_idcs += data_idcs[c][:take]
            data_idcs[c] = data_idcs[c][take:]

            budget -= take
            c = (c + 1) % n_labels

        clients_split += [(data[client_idcs], labels[client_idcs])]

    def print_split(clients_split):
        print("Data split:")
        for i, client in enumerate(clients_split):
            split = np.sum(client[1].reshape(1, -1) == np.arange(n_labels).reshape(-1, 1), axis=1)
            print(" - Client {}: {}".format(i, split))
        print()

    clients_split = np.array(clients_split)
    print_split(clients_split)
    return clients_split


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


def get_data_loaders_train(data_dir, nclients, batch_sizes, classes_pc=10, verbose=True,
                           transforms_train=None, dirichlet_level=0.1, transforms_eval=None, noniid=None):
    x_train, y_train, _, _ = get_cifar10(data_dir)

    if verbose:
        print_image_data_stats_train(x_train, y_train)

    split = None
    sample_number = None
    print('Non diid is ' + str(noniid))
    if noniid == 'quantity_skew':
        sample_proportions = generate_random_sample_proportions(n_clients=nclients, min_proportion=0.05,
                                                                max_proportion=0.50)

        split, sample_number = split_image_data_quantity(x_train, y_train, n_clients=nclients,
                                                         sample_proportions=sample_proportions,
                                                         verbose=verbose)
    elif noniid == 'dirichlet':
        split, sample_number = split_image_data_dirichlet(x_train, y_train, n_clients=nclients,
                                                          dirichlet_level=dirichlet_level,
                                                          verbose=verbose)

    elif noniid == 'label_skew':
        split = split_image_data(x_train, y_train, n_clients=nclients,
                                 classes_per_client=classes_pc, verbose=verbose)

    split_tmp = shuffle_list(split)

    client_loaders = [torch.utils.data.DataLoader(CustomImageDataset(x, y, transforms_train),
                                                  batch_size=bs, shuffle=True, drop_last=True) for (x, y), bs in zip(
        split_tmp, batch_sizes)]

    return client_loaders, sample_number


def get_data_loaders_test(data_dir, nclients, batch_size, classes_pc=10, verbose=True, transforms_train=None,
                          transforms_eval=None, noniid=None):
    _, _, x_test, y_test = get_cifar10(data_dir)

    if verbose:
        print_image_data_stats_test(x_test, y_test)

    test_loader = torch.utils.data.DataLoader(CustomImageDataset(x_test, y_test, transforms_eval),
                                              batch_size=batch_size, shuffle=False, )

    return test_loader
