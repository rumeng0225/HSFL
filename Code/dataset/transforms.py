from torchvision import transforms

from dataset.autoaugment import CIFAR10Policy


def get_transform(dataset):
    normalize_dict = {
        'tinyimagenet': ((0.4802, 0.4481, 0.3975), (0.2712, 0.2651, 0.2735)),
        'cifar10': ((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        'cifar100': ((0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762))
    }

    normalize_mean, normalize_std = normalize_dict.get(dataset, (None, None))

    if dataset in ['tinyimagenet']:
        train_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomCrop(64, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(normalize_mean, normalize_std),
            transforms.RandomErasing(p=0.5, scale=(0.125, 0.2)),
        ])
        val_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.ToTensor(),
            transforms.Normalize(normalize_mean, normalize_std),
        ])
    elif dataset in ['cifar10', 'cifar100']:
        train_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            CIFAR10Policy(),
            transforms.ToTensor(),
            transforms.Normalize(normalize_mean, normalize_std),
            transforms.RandomErasing(p=0.5, scale=(0.125, 0.2), ratio=(0.99, 1.0), value=0, inplace=False),
        ])
        val_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.ToTensor(),
            transforms.Normalize(normalize_mean, normalize_std),
        ])
    return train_transform, val_transform

