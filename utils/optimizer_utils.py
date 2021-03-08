import torch
from torch.optim import Adam
from torch.nn import DataParallel


def recreate_optimizer(optimizer):
    for state in optimizer.state.values():
        for k, v in state.items():
            if torch.is_tensor(v):
                state[k] = v.cuda()


def make_parallel(li_variables=None):

    if li_variables is None:
        return None

    parallel_variables = []

    for var in li_variables:
        var = DataParallel(var) if var else None
        parallel_variables.append(var)

    return parallel_variables


def make_cuda(li_variables=None):

    if li_variables is None:
        return None

    cuda_variables = []

    for var in li_variables:
        if not var is None:
            var = var.cuda()
        cuda_variables.append(var)

    return cuda_variables
