# coding=utf-8
from torch.utils.data import DataLoader

import config
from .cifar import CIFAR10, CIFAR100
from .noniid.cifar100_non_iid import *

__all__ = ['get_data_loader']

from .tiny_imagenet import TinyImageNet
from .noniid.tiny_imagenet_non_iid import get_data_loaders_train_tinet, get_data_loaders_test_tinet
from .transforms import get_transform


def get_data_loader(data_dir, batch_sizes=None, eval_batch_size=100, dataset='cifar10', num_workers=8, pin_memory=True,
                    num_clients=20, dirichlet=0.1, seed=11, noniid=''):
    assert batch_sizes is not None, "batch_sizes cannot be None during training"
    kwargs = {'num_workers': num_workers, 'pin_memory': pin_memory}

    print(f"INFO: Using {noniid} {dataset} dataset, batch size {batch_sizes}.")
    print(f'INFO: Creating {noniid} {dataset} train dataloader...')

    if dataset == 'tinyimagenet':
        data_dir = config.TINY_IMAGENET_PATH
        train_transform, val_transform = get_transform(dataset=dataset)

        if noniid in ['dirichlet']:
            train_loader, sample_number = get_data_loaders_train_tinet(data_dir,
                                                                       nclients=num_clients,
                                                                       batch_sizes=batch_sizes,
                                                                       verbose=True,
                                                                       transforms_train=train_transform,
                                                                       dirichlet_level=dirichlet,
                                                                       noniid=noniid)

            val_loader = get_data_loaders_test_tinet(data_dir, nclients=num_clients,
                                                     batch_sizes=eval_batch_size, verbose=True,
                                                     transforms_eval=val_transform,
                                                     noniid=noniid)
            return train_loader, sample_number, val_loader

        else:
            train_dataset = TinyImageNet(data_dir, transform=train_transform, train=True)
            images_per_client = int(len(train_dataset) / num_clients)
            print("INFO: Images per client is " + str(images_per_client))
            data_split = [images_per_client for _ in range(num_clients - 1)]
            data_split.append(len(train_dataset) - images_per_client * (num_clients - 1))
            sample_number = data_split

            traindata_split = torch.utils.data.random_split(train_dataset, data_split,
                                                            generator=torch.Generator().manual_seed(seed))

            train_loader = [torch.utils.data.DataLoader(x,
                                                        batch_size=bs,
                                                        shuffle=True,
                                                        drop_last=True,
                                                        sampler=None,
                                                        **kwargs) for x, bs in zip(traindata_split, batch_sizes)]

            val_dataset = TinyImageNet(data_dir, transform=val_transform, train=False)

            val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=eval_batch_size, shuffle=False,
                                                     **kwargs)
            return train_loader, sample_number, val_loader

    elif dataset == 'cifar10':
        data_dir = data_dir + '/cifar10'
        train_transform, val_transform = get_transform(dataset=dataset)

        if noniid in ['quantity_skew', 'dirichlet']:
            train_loader, sample_number = get_data_loaders_train(data_dir,
                                                                 nclients=num_clients,
                                                                 batch_sizes=batch_sizes,
                                                                 verbose=True,
                                                                 transforms_train=train_transform,
                                                                 dirichlet_level=dirichlet,
                                                                 noniid=noniid)

            val_loader = get_data_loaders_test(data_dir, nclients=num_clients,
                                               batch_size=eval_batch_size, verbose=True,
                                               transforms_eval=val_transform,
                                               noniid=noniid)
            return train_loader, sample_number, val_loader
        else:
            train_dataset = CIFAR10(data_dir, train=True, transform=train_transform,
                                    target_transform=None, download=True)

            images_per_client = int(train_dataset.data.shape[0] / num_clients)
            print("INFO: Images per client is " + str(images_per_client))

            data_split = [images_per_client for _ in range(num_clients - 1)]

            data_split.append(len(train_dataset) - images_per_client * (num_clients - 1))
            sample_number = data_split
            traindata_split = torch.utils.data.random_split(train_dataset, data_split,
                                                            generator=torch.Generator().manual_seed(seed))

            train_loader = [torch.utils.data.DataLoader(x,
                                                        batch_size=bs,
                                                        shuffle=True,
                                                        drop_last=True,
                                                        sampler=None,
                                                        **kwargs) for x, bs in zip(traindata_split, batch_sizes)]

            val_dataset = CIFAR10(data_dir, train=False, transform=val_transform,
                                  target_transform=None, download=True)
            val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=eval_batch_size, shuffle=False,
                                                     **kwargs)
            return train_loader, sample_number, val_loader
    elif dataset == 'cifar100':
        data_dir = data_dir + '/cifar100'
        train_transform, val_transform = get_transform(dataset=dataset)

        if noniid == 'quantity_skew':
            train_loader, sample_number = get_data_loaders_train_cf100(data_dir,
                                                                       nclients=num_clients,
                                                                       batch_sizes=batch_sizes,
                                                                       verbose=True,
                                                                       transforms_train=train_transform,
                                                                       dirichlet_level=dirichlet,
                                                                       noniid=noniid)

            val_loader = get_data_loaders_test_cf100(data_dir, nclients=num_clients,
                                                     batch_size=eval_batch_size, verbose=True,
                                                     transforms_eval=val_transform, noniid=noniid)
            return train_loader, sample_number, val_loader
        else:
            train_dataset = CIFAR100(data_dir, train=True, transform=train_transform,
                                     target_transform=None, download=True)

            images_per_client = int(train_dataset.data.shape[0] / num_clients)
            # print("Images per client is " + str(images_per_client))
            data_split = [images_per_client for _ in range(num_clients - 1)]

            data_split.append(len(train_dataset) - images_per_client * (num_clients - 1))
            sample_number = data_split
            traindata_split = torch.utils.data.random_split(train_dataset, data_split,
                                                            generator=torch.Generator().manual_seed(seed))
            train_loader = [torch.utils.data.DataLoader(x,
                                                        batch_size=bs,
                                                        shuffle=True,
                                                        drop_last=True,
                                                        sampler=None,
                                                        **kwargs) for x, bs in
                            zip(traindata_split, batch_sizes)]

            val_dataset = CIFAR100(data_dir, train=False,
                                   transform=val_transform,
                                   target_transform=None,
                                   download=True)

            val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=eval_batch_size, shuffle=False,
                                                     **kwargs)
            return train_loader, sample_number, val_loader
    else:
        raise NotImplementedError("The DataLoader for {} is not implemented.".format(dataset))
