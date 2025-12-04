import json
import shlex
import subprocess

import torch


def outlier_hook(module, input):
    assert isinstance(module, torch.nn.Linear)
    tracer = OutlierTracer.get_instance()
    hvalue = tracer.get_hvalue(module.weight)
    if hvalue not in tracer.hvalue2outlier_idx:
        outlier_idx = find_outlier_dims(module.weight)
        tracer.outliers.append(outlier_idx)
        tracer.hvalues.append(hvalue)
        if len(tracer.outliers) > 1:
            # assign the current layer the outlier idx found from the weight
            # of the previous linear layer
            if tracer.outliers[-1].numel() > 0:
                assert tracer.outliers[-1].max() < module.weight.shape[1]
            tracer.hvalue2outlier_idx[hvalue] = tracer.outliers[-1]

        else:
            # first layer, we cannot use the weight for outlier detection
            # we follow a mixed approach:
            # (1) zscore test of std of hidden dimension
            # (2) magnitude > 6 test
            merged = input[0].view(-1, input[0].shape[-1])
            # (1) zscore test of std of hidden dimension
            outlier_idx = find_outlier_dims(merged, reduction_dim=1, zscore=3)
            # (2) magnitude > 6 test
            dims = (torch.abs(input[0]) > 6).sum(dim=list(range(len(input[0].shape) - 1)))
            outlier_idx2 = torch.where(dims > 0)[0]
            outlier_idx = torch.cat([outlier_idx, outlier_idx2]).unique()
            tracer.hvalue2outlier_idx[hvalue] = outlier_idx
    else:
        for hook in tracer.hooks:
            hook.remove()


class OutlierTracer:
    _instance = None

    def __init__(self):
        raise RuntimeError("Call get_instance() instead")

    def initialize(self, model):
        self.last_w = None
        self.current_outlier_dims = None
        self.hvalues = []
        self.outliers = []
        self.hvalue2outlier_idx = {}
        self.initialized = True
        self.hooks = []

        for n, m in model.named_modules():
            if isinstance(m, torch.nn.Linear):
                self.hooks.append(m.register_forward_pre_hook(outlier_hook))

    def is_initialized(self):
        return getattr(self, "initialized", False)

    def get_hvalue(self, weight):
        return weight.data.storage().data_ptr()

    def get_outliers(self, weight):
        if not self.is_initialized():
            print("Outlier tracer is not initialized...")
            return None
        hvalue = self.get_hvalue(weight)
        if hvalue in self.hvalue2outlier_idx:
            return self.hvalue2outlier_idx[hvalue]
        else:
            return None

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = cls.__new__(cls)
        return cls._instance


def find_outlier_dims(weight, reduction_dim=0, zscore=4.0, topk=None, rdm=False):
    if rdm:
        return torch.randint(0, weight.shape[1], size=(topk,), device=weight.device).long()

    std = weight.std(reduction_dim)
    stdm = std.mean()
    stdstd = std.std()

    zstd = (std - stdm) / stdstd

    if topk is not None:
        _, idx = torch.topk(std.abs(), k=topk, dim=0)
    else:
        idx = torch.where(zstd > zscore)[0]

    return idx


def execute_and_return(command_string: str) -> tuple[str, str]:
    def _decode(subprocess_err_out_tuple):
        return tuple(to_decode.decode("UTF-8").strip() for to_decode in subprocess_err_out_tuple)

    def execute_and_return_decoded_std_streams(command_string):
        return _decode(
            subprocess.Popen(
                shlex.split(command_string),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            ).communicate(),
        )

    std_out, std_err = execute_and_return_decoded_std_streams(command_string)
    return std_out, std_err


def replace_linear(
    model,
    linear_replacement,
    skip_modules=("lm_head",),
    copy_weights=False,
    post_processing_function=None,
):
    """
    Replace linear modules with a new Linear module.
    Parameters:
        model (`torch.nn.Module`):
            Input model or `torch.nn.Module` as the function is run recursively.
        linear_replacement (`torch.nn.Module`):
            The linear module that replaces the old one. Only expects standard arguments.
            If other arguments need to be passed, use a lambda.
        skip_modules (`List[str]`, *optional*, defaults to `lm_head`):
            List of modules names not to convert. Defaults to `lm_head`.
        copy_weights (`bool`):
            Copy the weights from the old linear module to the new one
        post_processing_function (`str`):
            A function name of the replacement linear class that is called
            after processing.
    """
    for name, module in model.named_children():
        if len(list(module.children())) > 0:
            replace_linear(module, linear_replacement, skip_modules, copy_weights, post_processing_function)

        if isinstance(module, torch.nn.Linear) and name not in skip_modules:
            old_module = model._modules[name]
            model._modules[name] = linear_replacement(
                module.in_features,
                module.out_features,
                module.bias is not None,
            )
            if copy_weights:
                model._modules[name].weight = old_module.weight
                model._modules[name].bias = old_module.bias

            if post_processing_function is not None:
                func = getattr(module, post_processing_function, None)
                if func is not None:
                    func(module)
    return model


def pack_dict_to_tensor(source_dict):
    """
    Pack a dictionary into a torch tensor for storing quant_state items in state_dict.

    Parameters:
    - source_dict: The dictionary to be packed.

    Returns:
    A torch tensor containing the packed data.
    """
    json_str = json.dumps(source_dict)
    json_bytes = json_str.encode("utf-8")
    tensor_data = torch.tensor(list(json_bytes), dtype=torch.uint8)

    return tensor_data


def unpack_tensor_to_dict(tensor_data):
    """
    Unpack a torch tensor into a Python dictionary.

    Parameters:
    - tensor_data: The torch tensor containing the packed data.

    Returns:
    A Python dictionary containing the unpacked data.
    """
    json_bytes = bytes(tensor_data.cpu().numpy())
    json_str = json_bytes.decode("utf-8")
    unpacked_dict = json.loads(json_str)

    return unpacked_dict


LINEAR_8BIT_WEIGHTS_FORMAT_MAPPING = {"row": 0, "col32": 1, "col_turing": 2, "col_ampere": 3}
INVERSE_LINEAR_8BIT_WEIGHTS_FORMAT_MAPPING = {val: name for (name, val) in LINEAR_8BIT_WEIGHTS_FORMAT_MAPPING.items()}


def sync_gpu(t: torch.Tensor):
    if t.device.type == "cuda":
        torch.cuda.synchronize()
    elif t.device.type == "xpu":
        torch.xpu.synchronize()


class Conv1D_out(nn.Module):
    """
    1D-convolutional layer as defined by Radford et al. for OpenAI GPT (and also used in GPT-2).

    Basically works like a linear layer but the weights are transposed.

    Args:
        nf (`int`): The number of output features.
        nx (`int`): The number of input features.
    """

    def __init__(self, nf, nx):
        super().__init__()
        self.nf = nf
        self.nx = nx
        self.weight = nn.Parameter(torch.empty(nx, nf))
        self.bias = nn.Parameter(torch.zeros(nf))
        nn.init.normal_(self.weight, std=0.02)

    def __repr__(self) -> str:
        return "Conv1D(nf={nf}, nx={nx})".format(**self.__dict__)

    def forward(self, x):
        size_out = x.size()[:-1] + (self.nf,)
        x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        x = x.view(size_out)
        return x


def replace_conv1d_in_layers(model, layer_ids, target_path="mlp.c_fc", conv_cls=MyConv1D):
    """
    model.transformer.h[i].<target_path> 가 Conv1D인 경우만 교체
    """
    for i in layer_ids:
        block = model.transformer.h[i]

        # target_path: "mlp.c_fc" → ["mlp", "c_fc"]
        parts = target_path.split(".")
        parent = block
        for name in parts[:-1]:
            parent = getattr(parent, name)
        attr_name = parts[-1]

        old_conv = getattr(parent, attr_name)
        assert isinstance(old_conv, HFConv1D), f"{i}번째 레이어의 {target_path}는 Conv1D가 아님"

        nx, nf = old_conv.weight.shape
        new_conv = conv_cls(nf=nf, nx=nx)
        new_conv.weight.data.copy_(old_conv.weight.data)
        new_conv.bias.data.copy_(old_conv.bias.data)

        setattr(parent, attr_name, new_conv)
        print(f"[INFO] Replaced layer {i} {target_path} with {conv_cls.__name__}")
