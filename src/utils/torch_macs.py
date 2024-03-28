# pylint: skip-file

# Credits to Chillee: https://dev-discuss.pytorch.org/t/the-ideal-pytorch-flop-counter-with-torch-dispatch/505
from collections import defaultdict
from numbers import Number
from typing import Any, List

import torch
import torch.nn as nn
from torch.utils._python_dispatch import TorchDispatchMode
from torch.utils._pytree import tree_flatten, tree_map

aten = torch.ops.aten

def get_shape(i):
    return i.shape

def prod(x: List[int]) -> int:
    res: int = 1
    for i in x:
        res *= i
    return res

def matmul_macs(inputs: List[Any], outputs: List[Any]) -> Number:
    """
    Count macs for matmul.
    """
    # Inputs should be a list of length 2.
    # Inputs contains the shapes of two matrices.
    input_shapes = [get_shape(v) for v in inputs]
    #print("Matmul", input_shapes)
    assert len(input_shapes) == 2, input_shapes
    assert input_shapes[0][-1] == input_shapes[1][-2], input_shapes
    macs = prod(input_shapes[0]) * input_shapes[-1][-1]
    return macs

def mul(inputs: List[Any], outputs: List[Any]) -> Number:
    """
    Count macs for element-wise multiplication.
    """
    # Inputs should be a list of length 2.
    # Inputs contains the shapes of two matrices or 1 matrix and Number.
    # MACs should be the same as the number of elements in the output.
    output_shapes = [get_shape(v) for v in outputs]
    assert len(output_shapes) == 1, output_shapes
    macs = prod(output_shapes[0])
    return macs

def addmm_macs(inputs: List[Any], outputs: List[Any]) -> Number:
    """
    Count macs for fully connected layers.
    """
    # Count MAC for nn.Linear
    # inputs is a list of length 3.
    input_shapes = [get_shape(v) for v in inputs[1:3]]
    # input_shapes[0]: [batch size, input feature dimension]
    # input_shapes[1]: [batch size, output feature dimension]
    assert len(input_shapes[0]) == 2, input_shapes[0]
    assert len(input_shapes[1]) == 2, input_shapes[1]
    batch_size, input_dim = input_shapes[0]
    output_dim = input_shapes[1][1]
    macs = batch_size * input_dim * output_dim
    return macs

def bmm_macs(inputs: List[Any], outputs: List[Any]) -> Number:
    """
    Count macs for the bmm operation.
    """
    # Inputs should be a list of length 2.
    # Inputs contains the shapes of two tensor.
    assert len(inputs) == 2, len(inputs)
    input_shapes = [get_shape(v) for v in inputs]
    n, c, t = input_shapes[0]
    d = input_shapes[-1][-1]
    macs = n * c * t * d
    return macs


mac_mapping = {
    aten.mm: matmul_macs,
    aten.matmul: matmul_macs,
    aten.mul: mul,
    aten.addmm: addmm_macs,
    aten.bmm: bmm_macs,
}

def normalize_tuple(x):
    if not isinstance(x, tuple):
        return (x,)
    return x

class MACCounterMode(TorchDispatchMode):
    def __init__(self, module = None, debug=False):
        self.mac_counts = defaultdict(lambda: defaultdict(int))
        self.parents = ['Global']
        self.debug = debug
        list_of_modules = [] if module is None else [("", module)]
        while len(list_of_modules) > 0:
            name, module = list_of_modules.pop()
            children = dict(module.named_children()).items()
            if self.debug:
                print("Init:", name, len(children))
            if (len(children) == 0):
                module.register_forward_pre_hook(self.enter_module(name))
                module.register_forward_hook(self.exit_module(name))
            else:
                for name2, xmodule in children:
                    list_of_modules.append((name + "." + name2, xmodule))

    def enter_module(self, name):
        def f(module, inputs):
            self.parents.append(name)
            inputs = normalize_tuple(inputs)
            out = self.create_backwards_pop(name)(*inputs)
            return out

        return f

    def exit_module(self, name):
        def f(module, inputs, outputs):
            if self.debug:
                print("Exit mod", self.parents[-1], "-", name, ":", self.parents[-1] == name, "-", self.parents)
            #assert(self.parents[-1] == name)
            self.parents.pop()
            outputs = normalize_tuple(outputs)
            return self.create_backwards_push(name)(*outputs)
        return f

    def create_backwards_push(self, name):
        class PushState(torch.autograd.Function):
            @staticmethod
            def forward(ctx, *args):
                args = tree_map(lambda x: x.clone() if isinstance(x, torch.Tensor) else x, args)
                if len(args) == 1:
                    return args[0]
                return args

            @staticmethod
            def backward(ctx, *grad_outs):
                self.parents.append(name)
                return grad_outs

        return PushState.apply

    def create_backwards_pop(self, name):
        class PopState(torch.autograd.Function):
            @staticmethod
            def forward(ctx, *args):
                args = tree_map(lambda x: x.clone() if isinstance(x, torch.Tensor) else x, args)
                if len(args) == 1:
                    return args[0]
                return args

            @staticmethod
            def backward(ctx, *grad_outs):
                assert(self.parents[-1] == name)
                self.parents.pop()
                return grad_outs

        return PopState.apply

    def get_total(self) -> float:
        """
        Calculates and returns the total number of MACs (multiply–accumulate operations) for the given model.

        Returns:
            float: The total number of MACs in gigaMACs (GMACs).
        """
        total = sum(self.mac_counts['Global'].values())/1e9
        return total

    def __enter__(self):
        self.mac_counts.clear()
        super().__enter__()

    def __exit__(self, *args):
        print(f"Total: {sum(self.mac_counts['Global'].values())/1e9 } GMACS")
        for mod in self.mac_counts.keys():
            print(f"Module: ", mod)
            for k,v in self.mac_counts[mod].items():
                print(f"{k}: {v/1e9} GMACS")
            print()
        super().__exit__(*args)

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs if kwargs else {}

        out = func(*args, **kwargs)
        func_packet = func._overloadpacket
        if func_packet in mac_mapping:
            mac_count = mac_mapping[func_packet](args, normalize_tuple(out))
            if self.debug:
                print("Detect:", mac_count, "-", self.parents)
            for par in self.parents:
                self.mac_counts[par][func_packet] += mac_count
        else:
            if self.debug:
                print(f"Function {func_packet} not supported")
            else:
                pass
        return out
