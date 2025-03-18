from models.UNet_classify import UNet
from models.mobilevit import mobile_vit_xx_small
from models.resnet import ResNet18, ResNet34, ResNet50
from models.resnet_sl import ResNet18_client, ResNet18_server, ResNet34_client, ResNet34_server


def generate_model(model_name, args, device, train_layer=1, model_type='client'):
    if model_name == 'resnet18':
        model = ResNet18(num_classes=args.num_classes)
    elif model_name == 'resnet34':
        model = ResNet34(num_classes=args.num_classes)
    elif model_name == 'resnet50':
        model = ResNet50(num_classes=args.num_classes)
    elif model_name in ['resnet_18sl']:
        if model_type == "global":
            model = ResNet18(num_classes=args.num_classes)
        elif model_type == "client":
            model = ResNet18_client(num_classes=args.num_classes, train_layer=train_layer)
        elif model_type == "server":
            model = ResNet18_server(num_classes=args.num_classes, train_layer=train_layer)
    elif model_name in ['resnet_34sl']:
        if model_type == "global":
            model = ResNet34(num_classes=args.num_classes)
        elif model_type == "client":
            model = ResNet34_client(num_classes=args.num_classes, train_layer=train_layer)
        elif model_type == "server":
            model = ResNet34_server(num_classes=args.num_classes, train_layer=train_layer)
    elif model_name == "mobilevit":
        model = mobile_vit_xx_small(num_classes=args.num_classes)
    elif model_name == 'unet':
        model = UNet(num_classes=args.num_classes)

    return model.to(device)


