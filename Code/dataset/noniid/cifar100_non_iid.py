from torch.utils.data import DataLoader
from torchvision import transforms

from .cifar10_non_iid import *


def get_cifar100(datadir):
    data_train = torchvision.datasets.CIFAR100(datadir, train=True, download=True)
    data_test = torchvision.datasets.CIFAR100(datadir, train=False, download=True)

    x_train, y_train = data_train.data, np.array(data_train.targets)
    x_test, y_test = data_test.data, np.array(data_test.targets)

    return x_train, y_train, x_test, y_test


def split_image_data_quantity_cf100(data, labels, n_clients=100, sample_proportions=None, verbose=True):
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


def split_image_data_realwd_cf100(data, labels, n_clients=100, verbose=True):
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

    tmp = [np.random.randint(1, 100) for i in range(n_clients)]
    total_partition = sum(tmp)

    class_partition = break_into(total_partition, len(classes))

    class_partition = sorted(class_partition, reverse=True)
    class_partition_split = {}

    for ind, class_ in enumerate(classes):
        class_partition_split[class_] = [list(i) for i in np.array_split(label_indcs[ind], class_partition[ind])]

    # print([len(class_partition_split[key]) for key in  class_partition_split.keys()])

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

    sample_number = np.zeros(n_clients, dtype=int)

    def print_split(clients_split):
        print("Data split:")
        for i, client in enumerate(clients_split):
            split = np.sum(client[1].reshape(1, -1) == np.arange(n_labels).reshape(-1, 1), axis=1)
            sample_number[i] = np.sum(split)
            print(" - Client {}: {}".format(i, split))
        print()

    if verbose:
        print_split(clients_split)

    clients_split = np.array(clients_split)

    return clients_split, sample_number.tolist()


def get_default_data_transforms_cf100(train=True, verbose=True):
    transforms_train = {
        'cifar100': transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))]),
    }
    transforms_eval = {
        'cifar100': transforms.Compose([
            transforms.ToPILImage(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
    }
    if verbose:
        print("\nData preprocessing: ")
        for transformation in transforms_train['cifar100'].transforms:
            print(' -', transformation)
        print()

    return (transforms_train['cifar100'], transforms_eval['cifar100'])


def get_data_loaders_train_cf100(data_dir, nclients, batch_sizes, dirichlet_level=0.1,
                                 classes_pc=10, verbose=True,
                                 transforms_train=None,
                                 transforms_eval=None, noniid=None):
    x_train, y_train, _, _ = get_cifar100(data_dir)

    if verbose:
        print_image_data_stats_train(x_train, y_train)

    split = None
    sample_number = None
    print('Non diid is ' + str(noniid))
    if noniid == 'quantity_skew':
        sample_proportions = generate_random_sample_proportions(n_clients=5, min_proportion=0.05, max_proportion=0.20)
        split, sample_number = split_image_data_quantity_cf100(x_train, y_train,
                                                               n_clients=nclients,
                                                               sample_proportions=sample_proportions,
                                                               verbose=verbose)
    split_tmp = shuffle_list(split)

    client_loaders = [torch.utils.data.DataLoader(CustomImageDataset(x, y, transforms_train),
                                                  batch_size=bs, shuffle=True, drop_last=True) for (x, y), bs in zip(
        split_tmp, batch_sizes)]

    return client_loaders, sample_number


def get_data_loaders_test_cf100(data_dir, nclients, batch_size, classes_pc=10, verbose=True, transforms_train=None,
                                transforms_eval=None, noniid=None):
    _, _, x_test, y_test = get_cifar100(data_dir)

    if verbose:
        print_image_data_stats_test(x_test, y_test)

    test_loader = torch.utils.data.DataLoader(CustomImageDataset(x_test, y_test, transforms_eval),
                                              batch_size=batch_size, shuffle=False)

    return test_loader
