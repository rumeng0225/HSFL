from torchvision import datasets

if __name__ == '__main__':
    cifar_trainset = datasets.CIFAR100(root='../dataset', train=True, download=True)
    data = cifar_trainset.data / 255

    mean = data.mean(axis=(0, 1, 2))
    std = data.std(axis=(0, 1, 2))
    print(f"Mean : {mean} \n  STD: {std}")  # Mean : [0.491 0.482 0.446]   STD: [0.247 0.243 0.261]
