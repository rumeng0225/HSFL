import torch


def accuracy(output, target, topk=(1,)):
    """
    Computes the accuracy over the k top predictions for the specified values of k
    """
    if topk == (1,):
        topk = max(topk)
        with torch.no_grad():
            batch_size = target.size(0)
            pred_k = output.topk(topk, 1, True, True)[1]
            correct_k = pred_k.eq(target.view(-1, 1).expand_as(pred_k)).float().sum()
            acc = (correct_k * (100.0 / batch_size)).item()
        return acc
    else:
        with torch.no_grad():
            maxk = max(topk)
            batch_size = target.size(0)

            _, pred = output.topk(maxk, 1, True, True)
            pred = pred.t()
            correct = pred.eq(target.view(1, -1).expand_as(pred))

            res = []
            for k in topk:
                correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
                res.append(correct_k.mul_(100.0 / batch_size).item())

            return res


def get_the_number_of_params(model, is_trainable=False):
    """
    get the number of the model
    """
    if is_trainable:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    return sum(p.numel() for p in model.parameters())
