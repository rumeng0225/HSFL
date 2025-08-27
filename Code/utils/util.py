import copy
import json
import os
import shutil
from collections.abc import Iterable

import torch
from torch import nn
from torchstat import stat

from models.resnet import BasicBlock, ResNet
from utils.summary import summary


def Average(lst):
    return sum(lst) / len(lst)


def copy_layers(up_to, server_model_dict, client_model_dict):
    for server_key, server_tensor in server_model_dict.items():
        for i in range(up_to):
            layer_name = f'layer{i}'
            if layer_name in server_key:
                client_key = server_key.replace("server_model", "client_model")
                client_tensor = copy.deepcopy(server_tensor)
                client_model_dict[client_key] = client_tensor


def clever_format(nums, format="%.2f"):
    if not isinstance(nums, Iterable):
        nums = [nums]
    clever_nums = []

    for num in nums:
        if num > 1e12:
            clever_nums.append(format % (num / 1e12) + "T")
        elif num > 1e9:
            clever_nums.append(format % (num / 1e9) + "G")
        elif num > 1e6:
            clever_nums.append(format % (num / 1e6) + "M")
        elif num > 1e3:
            clever_nums.append(format % (num / 1e3) + "K")
        else:
            clever_nums.append(format % num + "B")

    clever_nums = clever_nums[0] if len(clever_nums) == 1 else (*clever_nums,)

    return clever_nums


def print_grad(net):
    for name, parms in net.named_parameters():
        print('-->name:', name, '-->grad_requirs:', parms.requires_grad, ' -->grad_value:', parms.grad)


def save_time_to_file(file_name, client_id, train_dict):
    file_name += 'time/'
    os.makedirs(file_name, exist_ok=True)
    filename = os.path.join(file_name, 'client_' + str(client_id) + '.json')

    with open(filename, 'w') as f:
        json.dump(train_dict, f, indent=4)


def save2tensorbord_file(val_writer, tag_list, data_list, global_step):
    for i in range(len(tag_list)):
        val_writer.add_scalar(tag_list[i], data_list[i], global_step)


def save_modeldict(model, path):
    state_dict = model.state_dict()

    with open(path, 'w') as file:
        file.write(str(state_dict))
        file.close()


def save_checkpoint(state, is_best, model_dir, filename='checkpoint.pth.tar'):
    model_fname = os.path.join(model_dir, filename)
    torch.save(state, model_fname)
    if is_best:
        shutil.copyfile(model_fname,
                        os.path.join(model_dir, 'model_best.pth.tar'))


def stat_net(model, arch, input_shape=224, is_save=False, summary_type=1):
    if summary_type == 1:
        summary(model, torch.rand((1, 3, input_shape, input_shape)),
                target=torch.ones(1, dtype=torch.long),
                is_splitnet=False,
                is_save=is_save,
                arch=arch)
    elif summary_type == 2:
        stat(model, (3, input_shape, input_shape))
    elif summary_type == 3:
        def my_hook(Module, input, output):
            outshapes.append(output.shape)
            modules.append(Module)

        names, modules, outshapes = [], [], []
        for name, m in model.named_modules():
            if isinstance(m, nn.Conv2d):
                m.register_forward_hook(my_hook)
                names.append(name)

        input = torch.rand(1, 3, input_shape, input_shape)
        y = model(input)

        total_para_nums = 0
        total_flops = 0
        for i, m in enumerate(modules):
            Cin = m.in_channels
            Cout = m.out_channels
            k = m.kernel_size
            g = m.groups
            Hout = outshapes[i][2]
            Wout = outshapes[i][3]
            if m.bias is None:
                para_nums = k[0] * k[1] * Cin / g * Cout

                flops = (k[0] * k[1] * Cin / g - 1) * Cout * Hout * Wout
                smashed_size = Hout * Wout * Cout * 4
            else:
                para_nums = (k[0] * k[1] * Cin / g + 1) * Cout

                flops = k[0] * k[1] * Cin / g * Cout * Hout * Wout
                smashed_size = Hout * Wout * Cout * 4

            para_nums = int(para_nums)
            flops = int(flops)
            smashed_size = int(smashed_size)
            print(names[i], 'para(M):', para_nums / 1e6, 'flops(M):', flops / 1e6, 'smashed_size:', smashed_size)
            total_para_nums += para_nums
            total_flops += flops
        print('total conv parameters(M):', total_para_nums / 1e6, 'total conv FLOPs(M):', total_flops / 1e6)
        return total_para_nums, total_flops


if __name__ == '__main__':
    layers, block = [2, 2, 2, 2], BasicBlock
    model = ResNet(block=block, layers=layers, num_classes=10)
    print(model)
    # torch.save(model, 'model.pth')

    stat_net(model, arch='resnet18', input_shape=224, is_save=False, summary_type=1)
    exit(0)
