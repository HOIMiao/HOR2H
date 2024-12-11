import torch
from typing import Optional

from torch import nn
from torch.overrides import (
    has_torch_function, has_torch_function_unary, has_torch_function_variadic,
    handle_torch_function)
from torch.nn import _reduction as _Reduction
import warnings


Tensor = torch.Tensor

def maxmse_loss(
    input: Tensor,
    target: Tensor,
    size_average: Optional[bool] = None,
    reduce: Optional[bool] = None,
    reduction: str = "mean",
) -> Tensor:  # noqa: D400,D402
    r"""mse_loss(input, target, size_average=None, reduce=None, reduction='mean') -> Tensor

    Measures the element-wise mean squared error.
    See :class:`~torch.nn.MSELoss` for details.
    """
    if has_torch_function_variadic(input, target):
        return handle_torch_function(
            maxmse_loss, (input, target), input, target, size_average=size_average, reduce=reduce, reduction=reduction
        )
    if not (target.size() == input.size()):
        warnings.warn(
            f"Using a target size ({target.size()}) that is different to the input size ({input.size()}). "
            "This will likely lead to incorrect results due to broadcasting. "
            "Please ensure they have the same size.",
            stacklevel=2,
        )
    if size_average is not None or reduce is not None:
        reduction = _Reduction.legacy_get_string(size_average, reduce)

    expanded_input, expanded_target = torch.broadcast_tensors(input, target)
    # return torch._C._nn.mse_loss(expanded_input, expanded_target, _Reduction.get_enum(reduction))

    criterion1 = nn.MSELoss(reduction="sum")    #reduction='sum'，求所有对应位置的差的平方的和 是一个标量
    result_sum = criterion1(expanded_target, expanded_input) #误差的平方的和
    criterion2 = nn.MSELoss(reduction='none')   #reduction='none'，求所有对应位置的差的平方 是一个矩阵
    result = criterion2(expanded_target, expanded_input)    #误差的平方
    # result = expanded_input - expanded_target
    # result_sum = result.sum()
    # w_i = result / result_sum   #权重
    w_i = torch.divide(result, result_sum)
    # result = result * w_i
    result = torch.mul(w_i, result)
    result = result.sum()
    return result

