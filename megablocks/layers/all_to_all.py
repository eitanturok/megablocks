# Copyright 2024 Databricks
# SPDX-License-Identifier: Apache-2.0
from typing import Any, Tuple, Optional

import torch
import torch.distributed as dist


class AllToAllOp(torch.autograd.Function):

    @staticmethod
    def forward(ctx: Any, x: torch.Tensor, output_split_sizes: list[int], input_split_sizes:list[int], group: Optional[dist.ProcessGroup], async_op: bool):
        out = torch.empty((sum(output_split_sizes),) + x.shape[1:], device=x.device, dtype=x.dtype)

        ctx.input_shape = x.shape
        ctx.output_split_sizes = output_split_sizes
        ctx.input_split_sizes = input_split_sizes
        ctx.group = group
        handle = dist.all_to_all_single(
            out,
            x,
            output_split_sizes=output_split_sizes,
            input_split_sizes=input_split_sizes,
            group=group,
            async_op=async_op,
        )
        return out, handle

    @staticmethod
    def backward(ctx: Any, grad: torch.Tensor, _):
        if ctx.needs_input_grad[0]:
            out = torch.empty(
                ctx.input_shape,
                device=grad.device,
                dtype=grad.dtype,
            )
            dist.all_to_all_single(
                out,
                grad,
                output_split_sizes=ctx.input_split_sizes,
                input_split_sizes=ctx.output_split_sizes,
                group=ctx.group,
            )
            return out, None, None, None, None
        return None, None, None, None, None



def all_to_all(x: torch.Tensor, output_split_sizes: list[int], input_split_sizes:list[int], group: Optional[dist.ProcessGroup], async_op: bool=False) -> Tuple[torch.Tensor, ...]:
    output = AllToAllOp.apply(
        x,
        output_split_sizes,
        input_split_sizes,
        group,
        async_op,
    )
    assert output is not None
    return output
