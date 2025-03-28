# -*- coding: utf-8 -*-
# @Author  : DevinYang(pistonyang@gmail.com)
__all__ = ['summary']

import csv
import os
from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn


def _flops_str(flops):
    preset = [(1e12, 'T'), (1e9, 'G'), (1e6, 'M'), (1e3, 'K')]

    for p in preset:
        if flops // p[0] > 0:
            N = flops / p[0]
            ret = "%.1f%s" % (N, p[1])
            return ret
    ret = "%.1f" % flops
    return ret


def _cac_grad_params(p, w):
    t, n = 0, 0
    if w.requires_grad:
        t += p
    else:
        n += p
    return t, n


def _cac_conv(layer, input, output):
    # bs, ic, ih, iw = input[0].shape
    oh, ow = output.shape[-2:]
    kh, kw = layer.kernel_size
    ic, oc = layer.in_channels, layer.out_channels
    g = layer.groups

    tb_params = 0
    ntb__params = 0
    flops = 0
    if hasattr(layer, 'weight') and hasattr(layer.weight, 'shape'):
        params = np.prod(layer.weight.shape)
        t, n = _cac_grad_params(params, layer.weight)
        tb_params += t
        ntb__params += n
        flops += (2 * ic * kh * kw - 1) * oh * ow * (oc // g)
    if hasattr(layer, 'bias') and hasattr(layer.bias, 'shape'):
        params = np.prod(layer.bias.shape)
        t, n = _cac_grad_params(params, layer.bias)
        tb_params += t
        ntb__params += n
        flops += oh * ow * (oc // g)
    return tb_params, ntb__params, flops


def _cac_xx_norm(layer, input, output):
    tb_params = 0
    ntb__params = 0
    if hasattr(layer, 'weight') and hasattr(layer.weight, 'shape'):
        params = np.prod(layer.weight.shape)
        t, n = _cac_grad_params(params, layer.weight)
        tb_params += t
        ntb__params += n
    if hasattr(layer, 'bias') and hasattr(layer.bias, 'shape'):
        params = np.prod(layer.bias.shape)
        t, n = _cac_grad_params(params, layer.bias)
        tb_params += t
        ntb__params += n
    if hasattr(layer, 'running_mean') and hasattr(layer.running_mean, 'shape'):
        params = np.prod(layer.running_mean.shape)
        ntb__params += params
    if hasattr(layer, 'running_var') and hasattr(layer.running_var, 'shape'):
        params = np.prod(layer.running_var.shape)
        ntb__params += params
    in_shape = input[0]
    flops = np.prod(in_shape.shape)
    if layer.affine:
        flops *= 2
    return tb_params, ntb__params, flops


def _cac_linear(layer, input, output):
    ic, oc = layer.in_features, layer.out_features

    tb_params = 0
    ntb__params = 0
    flops = 0
    if hasattr(layer, 'weight') and hasattr(layer.weight, 'shape'):
        params = np.prod(layer.weight.shape)
        t, n = _cac_grad_params(params, layer.weight)
        tb_params += t
        ntb__params += n
        flops += (2 * ic - 1) * oc
    if hasattr(layer, 'bias') and hasattr(layer.bias, 'shape'):
        params = np.prod(layer.bias.shape)
        t, n = _cac_grad_params(params, layer.bias)
        tb_params += t
        ntb__params += n
        flops += oc
    return tb_params, ntb__params, flops


@torch.no_grad()
def summary(model, x, target, return_results=False, is_splitnet=True, is_save=False, arch='net'):
    """
    Args:
        model (nn.Module): model to summary
        x (torch.Tensor): input data
        return_results (bool): return results
    Returns:
    """

    model.eval()

    def register_hook(layer):

        def hook(layer, input, output):
            model_name = str(layer.__class__.__name__)
            module_idx = len(model_summary)
            s_key = '{}-{}'.format(model_name, module_idx + 1)
            model_summary[s_key] = OrderedDict()
            model_summary[s_key]['input_shape'] = list(input[0].shape)
            if isinstance(output, (tuple, list)):
                model_summary[s_key]['output_shape'] = [
                    list(o.shape) for o in output
                ]
            else:
                model_summary[s_key]['output_shape'] = list(output.shape)
            tb_params = 0
            ntb__params = 0
            flops = 0

            if isinstance(layer, nn.Conv2d):
                tb_params, ntb__params, flops = _cac_conv(layer, input, output)
            elif isinstance(layer, (nn.BatchNorm2d, nn.GroupNorm)):
                tb_params, ntb__params, flops = _cac_xx_norm(
                    layer, input, output)
            elif isinstance(layer, nn.Linear):
                tb_params, ntb__params, flops = _cac_linear(
                    layer, input, output)

            model_summary[s_key]['trainable_params'] = tb_params
            model_summary[s_key]['non_trainable_params'] = ntb__params
            model_summary[s_key]['params'] = tb_params + ntb__params
            model_summary[s_key]['flops'] = flops

        # if not isinstance(layer, (nn.Sequential, nn.ModuleList,
        #                           nn.Identity, nn.ModuleDict,
        #                           nn.ReLU,nn.MaxPool2d)):
        #     hooks.append(layer.register_forward_hook(hook))
        if isinstance(layer, (nn.Conv2d,
                              nn.BatchNorm2d,
                              # nn.LayerNorm,
                              nn.Linear)):
            hooks.append(layer.register_forward_hook(hook))

    model_summary = OrderedDict()
    hooks = []
    model.apply(register_hook)

    if is_splitnet:
        model(x, target=target, mode='summary')
    else:
        model(x)

    for h in hooks:
        h.remove()

    print('-' * 100)
    line_new = "{:>20}  {:>25} {:>15} {:>15} {:>15}".format(
        "Layer (type)", "Output Shape", "Params", "FLOPs(M)", "FLOPs(M+A) #")
    print(line_new)
    print('=' * 100)
    total_params = 0
    trainable_params = 0
    total_flops = 0.
    for layer in model_summary:
        line_new = "{:>20}  {:>25} {:>15} {:>15} {:>15}".format(
            layer,
            str(model_summary[layer]['output_shape']),
            model_summary[layer]['params'],
            model_summary[layer]['flops'] // 2,
            model_summary[layer]['flops'],
        )
        print(line_new)
        total_params += model_summary[layer]['params']
        trainable_params += model_summary[layer]['trainable_params']
        total_flops += model_summary[layer]['flops']

    param_str = _flops_str(total_params)
    flop_str = _flops_str(total_flops)
    flop_str_m = _flops_str(total_flops // 2)
    param_size = total_params * 4 / (1024 ** 2)
    print('=' * 100)
    print('        Total parameters: {:,}  {}'.format(total_params, param_str))
    print('    Trainable parameters: {:,}'.format(trainable_params))
    print(
        'Non-trainable parameters: {:,}'.format(total_params - trainable_params))
    print('Total flops(M)  : {:,}  {}'.format(total_flops // 2, flop_str_m))
    print('Total flops(M+A): {:,}  {}'.format(total_flops, flop_str))
    print('-' * 100)
    print('Parameters size (MB): {:.2f}'.format(param_size))

    if is_save:
        save_net_info_to_csv(arch, model_summary)

    if return_results:
        return total_params, total_flops


def save_net_info_to_csv(arch, model_summary):

    csv_file_path = str(os.getcwd())+'\\' + f'{arch}.csv'

    data_list = []
    for index, layer in enumerate(model_summary):
        layer_info = model_summary[layer]
        layer_name = layer
        output_shape = layer_info['output_shape']
        channel = output_shape[1]

        if 'Linear' in layer_name:
            shape = 1
        else:
            shape = output_shape[2]
        params = layer_info['params'],
        model_flops = layer_info['flops'] // 2,
        model_flops_MpA = layer_info['flops'],

        layer_dict = {
            "": index + 1,
            "Layer": layer_name.split('-')[0],
            "Channel": channel,
            "Shape": shape,
            "Params": params[0],
            "FLOPs(M)": model_flops[0],
            "FLOPs(M+A)": model_flops_MpA[0],
        }
        data_list.append(layer_dict)

    with open(csv_file_path, mode='w', newline='') as file:
        fieldnames = data_list[0].keys()
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for row in data_list:
            writer.writerow(row)


def save_model_to_json(args, model_files, server_models=None, mode="w"):
    """
    Save model to a json file
	"""
    os.makedirs(args.model_dir, exist_ok=True)

    filename = os.path.join(args.model_dir, 'model' + '.txt')
    for i, model_file in enumerate(model_files):
        filename = os.path.join(args.model_dir, 'model_' + str(i) + '.txt')
        # model_file = vars(model_file)
        with open(filename, mode) as f:
            f.write(str(model_file))
            if server_models is not None:
                f.write("\n========================================================\n")
                f.write(str(server_models[i]))


if __name__ == '__main__':
    A = nn.Conv2d(50, 10, 3, padding=1, groups=5, bias=True)
    summary(A, torch.rand((1, 50, 10, 10)),
            target=torch.ones(1, dtype=torch.long),
            is_splitnet=False)
    for name, p in A.named_parameters():
        print(name, p.size())
