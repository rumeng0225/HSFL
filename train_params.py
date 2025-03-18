import config


def add_parser_params(parser):
    parser.add_argument('--data', type=str, metavar='DIR', default=str(config.data_dir) + "/data/",
                        help='path to dataset')

    parser.add_argument('--seed', default=1, type=int, help='seed for initializing training')

    parser.add_argument('--display', help='display in controller', action='store_true')

    parser.add_argument('--arch', type=str, default='resnet18', choices=['resnet18', 'resnet34', 'resnet50',
                                                                         'resnet18sl', 'resnet34sl',
                                                                         'mobilevit', 'unet', ],
                        help='The name of the neural architecture')

    parser.add_argument('--total_round', default=300, type=int, metavar='N',
                        help='number of total epochs to run (default: 300)')

    parser.add_argument('--algo_name', type=str, default='FedAvg', choices=['FedAvg', 'FedProx', 'FedAvg_SFL'],
                        help='The name of the used algorithm')

    parser.add_argument('--device', default=0, nargs='+', type=int, help='GPU ID, -1 for CPU (default: 0).')

    parser.add_argument('--workers', default=0, type=int, metavar='N',
                        help='number of data loading workers (default: 0)')

    # parser.add_argument('--model_dir', type=str, default="/log",
    #                     help='dir to which model is saved (default: ./model_dir)')

    parser.add_argument('--eval_per_epoch', default=1, type=int, help='run evaluation per eval_per_epoch')

    parser.add_argument('--batch_size', default=[32], type=list, metavar='N', help='mini-batch size (default: 32)')

    parser.add_argument('--eval_batch_size', default=100, type=int, metavar='N',
                        help='mini-batch size (default: 100), this is will not be divided by the number of gpus.')

    parser.add_argument('--padding', default=4, type=int, metavar='N', help='padding size (default: 4)')

    # learning rate
    parser.add_argument('--lr', '--learning_rate', default=0.01, type=float, metavar='LR',
                        help='initial learning rate (default: 0.01)', dest='lr')

    parser.add_argument('--end_lr', default=1e-4, type=float,
                        help='The ending learning rate.')

    parser.add_argument('--optimizer', type=str, default='SGD', choices=['SGD', 'AdamW'],
                        help='The optimizer.')

    parser.add_argument('--scheduler', type=str, default='CosineAnnealingLR', choices=['CosineAnnealingLR', 'None'])

    # parameters of the optimizer
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M', help='optimizer momentum (default: 0.9)')

    parser.add_argument('--is_nesterov', default=1, type=int, help='using Nesterov accelerated gradient or not')

    # setting about the weight decay
    parser.add_argument('--weight_decay', default=1e-4, type=float, metavar='W', help='weight decay (default: 1e-4)',
                        dest='weight_decay')

    parser.add_argument('--evaluate', dest='evaluate', action='store_true', help='evaluate model on validation set')

    # datatset
    parser.add_argument('--dataset', type=str, default='cifar10',
                        choices=['cifar10', 'cifar100', 'tinyimagenet'], help='dataset name (default: cifar10)')

    parser.add_argument('--sampler', default='iid', type=str, choices=['iid', 'dirichlet', 'quantity_skew'],
                        help='Whether noniid.')

    parser.add_argument('--dirichlet', default=0.5, type=int,
                        help='The parameter for the dirichlet distribution for data partitioning, when noniid type is dirichlet')

    # federated learning
    parser.add_argument('--num_clients', default=20, type=int, help='Whether training federated learning or not.')

    parser.add_argument('--frac', default=1.0, type=float, help='The fraction of clients in a round.')

    # for FedProx
    parser.add_argument('--prox_mu', default=0.01, type=int,
                        help='The hypter parameter for the FedProx (default: 0.01).')

    parser.add_argument('--local_epoch', default=1, type=int, help='Local epoch of each client.')

    args = parser.parse_args()

    num_classes_dict = {
        'cifar10': 10,
        'cifar100': 100,
        'tinyimagenet': 200
    }

    args.num_classes = num_classes_dict[args.dataset]

    return args
