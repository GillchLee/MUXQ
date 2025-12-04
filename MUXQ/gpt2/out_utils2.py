import torch
import torch.nn as nn
import pdb
import numpy as np

import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  
from matplotlib import colors


def activation_outlier_hook(module, input):
    """
    Linear layer의 input activation
    outlier hidden-dim(채널) index finding hook
    - zscore  + |x| > out_mag
    """
    assert isinstance(module, MixedInputInt8Conv1D)
    tracer = OutlierTracer.get_instance()
    hvalue = tracer.get_hvalue(module.weight)

    x = input[0] 
    merged = x.view(-1, x.shape[-1])  # (N, H)


    ######################################################################
    #############           Adjust Outlier parameters       ##############
    #####################################################################

    
    ## 1) zscore test: (std over batch)
    
    outlier_idx_z = find_outlier_dims(
        merged,
        reduction_dim=0,
        zscore=3.0,   # similar value with llm.int8()
        topk=None,
    )

    # 2) define outlier condition. magnitude > out_mag, test: outlier
    dims = (merged.abs() > 5 ).sum(dim=0)   # shape (H,)
    outlier_idx_mag = torch.where(dims > 0)[0]

    ######################################################################
    ############# Normal Quantization without any technique ##############
    ######################################################################
    
    # outlier_idx_z = find_outlier_dims(
    #     merged,
    #     reduction_dim=0,
    #     zscore=1000.0,
    #     topk=None,
    # )
    # dims = (merged.abs() > 1000 ).sum(dim=0)   # shape (H,)
    # outlier_idx_mag = torch.where(dims > 0)[0]

    ######################################################################
    ######################################################################
    ######################################################################
    
    
    outlier_idx = torch.cat([outlier_idx_z, outlier_idx_mag]).unique()

    if hvalue in tracer.hvalue2outlier_idx:
        prev = tracer.hvalue2outlier_idx[hvalue].to(outlier_idx.device)
        outlier_idx = torch.unique(torch.cat([prev, outlier_idx]))

    tracer.hvalue2outlier_idx[hvalue] = outlier_idx.detach().cpu()

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
            if isinstance(m, MixedInputInt8Conv1D):
#                print(f"[Tracer] hook on module: {n}")  # print module name
                self.hooks.append(m.register_forward_pre_hook(activation_outlier_hook))

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





# ---------- Quant helpers (symmetric, zp=0) ----------
@torch.no_grad()
def per_row_absmax_scale(x2d, qmax, eps=1e-12):
    # x2d: (N, K) -> row-wise scale (N,)
    m = x2d.abs().amax(dim=1).clamp_min(eps)   # (N,)
    s = m / qmax                                # (N,)
    return s

@torch.no_grad()
def quantize_int8_per_row(x2d, eps=1e-12):
    # x2d: (N, K)
    qmax = 2**7 - 1  # 127
    s = per_row_absmax_scale(x2d, qmax, eps)    # (N,)
    xq = torch.round(x2d / s.view(-1, 1)).clamp_(-qmax-1, qmax).to(torch.int8)
    return xq , s                                 # xq:(N,K int8), s:(N,)

@torch.no_grad()
def quantize_int16_per_row(x2d, eps=1e-12):
    # x2d: (N, K)
    qmax = 2**15 - 1  # 127
    s = per_row_absmax_scale(x2d, qmax, eps)    # (N,)
    xq = torch.round(x2d / s.view(-1, 1)).clamp_(-qmax-1, qmax).to(torch.int8)
    return xq , s                                 # xq:(N,K int8), s:(N,)

# ---------- Quant helpers for per-channel INT8 (symmetric, zp=0) ----------

@torch.no_grad()
def per_channel_absmax_scale(x2d, qmax, eps=1e-12):
    """
    x2d: (N, K)
    return: scale for each channel K -> shape (K,)
    """
    # dim=0 or dim=1?   # (N, K)에서 column-wise = dim=0 기준 max(abs)
    m = x2d.abs().amax(dim=0).clamp_min(eps)   # (K,)
    s = m / qmax                               # (K,)
    return s


@torch.no_grad()
def quantize_int8_per_channel(x2d, eps=1e-12):
    """
    x2d: (N, K)
    returns:
        xq: (N, K) int8
        s:  (K,) per-channel scale
    """
    qmax = 2**7 - 1  # 127
    s = per_channel_absmax_scale(x2d, qmax, eps)   # (K,)
    xq = torch.round(x2d / s.view(1, -1)).clamp_(-qmax-1, qmax).to(torch.int8)
    return xq, s

@torch.no_grad()
def quantize_int16_per_row(x2d, eps=1e-12):
    # x2d: (N, K_sel)  -> int16 양자화
    qmax = 2**15 - 1  # 32767
    s = per_row_absmax_scale(x2d, qmax, eps)    # (N,)
    xq16 = torch.round(x2d / s.view(-1, 1)).clamp_(-qmax-1, qmax).to(torch.int16)
    return xq16, s                               # xq16:(N,K_sel int16), s:(N,)

@torch.no_grad()
def quantize_int16_per_row(x2d, eps=1e-12):
    # x2d: (N, K_sel)  -> int16 양자화
    qmax = 2**15 - 1  # 32767
    s = per_channel_absmax_scale(x2d, qmax, eps)    # (N,)
    xq16 = torch.round(x2d / s.view(1, -1)).clamp_(-qmax-1, qmax).to(torch.int16)
    return xq16, s       


class MixedInputInt8Conv1D(nn.Module):
    """

    """
    def __init__(self, nf, nx, selected_cols=None, act_bits_non=8, w_bits=8):
        super().__init__()
        assert act_bits_non == 8, 
        assert w_bits == 8, 
        self.nf = nf
        self.nx = nx
        self.selected_cols = sorted(set(selected_cols or []))
        mask = torch.zeros(nx, dtype=torch.bool)
        if len(self.selected_cols) > 0:
            mask[self.selected_cols] = True
        self.register_buffer("sel_mask", mask, persistent=False)
        self.register_buffer("non_mask", ~mask, persistent=False)
        self.register_buffer("sel_idx", torch.tensor(self.selected_cols, dtype=torch.long), persistent=False)
        self.register_buffer("non_idx", torch.nonzero(~mask, as_tuple=False).flatten(), persistent=False)

        # FP weights / bias
        self.weight = nn.Parameter(torch.empty(nx, nf))
        self.bias   = nn.Parameter(torch.zeros(nf))
        nn.init.normal_(self.weight, std=0.02)

        # Quantized weight buffers
        self.register_buffer("W_int8", None, persistent=False)   # (nx, nf), int8
        self.register_buffer("W_scale", None, persistent=False)  # (nf,), float



    @torch.no_grad()
    def _set_selected_cols_from_idx(self, idx: torch.Tensor):
        """
        idx: 1D LongTensor, outlier tracer가 반환한 column index
        """
        if idx is None or idx.numel() == 0:
            self.selected_cols = []
            mask = torch.zeros(self.nx, dtype=torch.bool, device=self.weight.device)
        else:
            idx = idx.to(self.weight.device).long().unique()
            idx = idx[(idx >= 0) & (idx < self.nx)] 
            self.selected_cols = sorted(idx.tolist())
            mask = torch.zeros(self.nx, dtype=torch.bool, device=self.weight.device)
            mask[self.selected_cols] = True

        non_mask = ~mask

        self.sel_mask.data.resize_(self.nx)
        self.sel_mask.data.copy_(mask)

        self.non_mask.data.resize_(self.nx)
        self.non_mask.data.copy_(non_mask)

        self.sel_idx = idx
        self.non_idx = torch.nonzero(non_mask, as_tuple=False).flatten()

    @torch.no_grad()
    def update_selected_cols_from_tracer(self, tracer=None):

        try:
            from your_tracer_module import OutlierTracer  # 실제 경로로 수정
        except ImportError:
            return  # tracer 모듈이 없으면 조용히 패스

        tracer = tracer or OutlierTracer.get_instance()
        if not tracer.is_initialized():
            return

        out_idx = tracer.get_outliers(self.weight)
        if out_idx is None:
            return

        self._set_selected_cols_from_idx(out_idx)



    @torch.no_grad()
    def prepare(self): # Weight quantization 
        # per-channel (nf=dim=0, weight shape (nx, nf))
        qmax = 2**7 - 1  # Weight Quantization bit 
        m = self.weight.abs().amax(dim=0, keepdim=True).clamp_min(1e-12)  # (1, nf)
        s = (m / qmax).squeeze(0)                                         # (nf,)
        wq = torch.round(self.weight / s.view(1, -1)).clamp_(-qmax-1, qmax).to(torch.int8)
        self.W_int8 = wq
        self.W_scale = s
        self.W_fp = self.weight

    @staticmethod
    @torch.no_grad()
    def split_outlier_columns_fp(x2d, out_idx):
        """
        x2d   : (N, K) float
        out_idx: (Ks,) long, outlier column indices in [0, K)
        return:
            x_main: (N, K)   # outlier shifted by alpha
            x_out : (N, Ks)  # outlier shifted by beta
        """
        if out_idx is None or len(out_idx) == 0:
            # if no outlier, return original value
            return x2d, None

        x_main = x2d.clone()
        
        ## split by shifting exponent -> we implement as multiply unit to find similar results

        x_out               =   x_main[:, out_idx] * 0.25  # * 0.5 => alpha =1 , * 0.25=> alpha=2 , * 0.125 => alpha = 3
        x_main[:, out_idx]  =   x_main[:, out_idx] * 0.25  # * 0.5 => beta =1 , *0.25 => beta=2 , * 0.125 => beta = 3     
        
        return x_main, x_out

    @staticmethod
    @torch.no_grad()
    def split_outlier_columns(x2d, out_idx): # to implement LLM.int8() quantization method
        # masking x main, assign outlier columns to x out 

        if out_idx is None or len(out_idx) == 0:
            return x2d, None

        x_main = x2d.clone()

        x_out = x_main[:, out_idx]   # (N, Ks)
        x_main[:, out_idx] = 0 

        return x_main, x_out

    def forward(self, x):
        if not self.selected_cols:  # only selected_cols is empty
            try:
                tracer = OutlierTracer.get_instance()
                if tracer.is_initialized():
                    out_idx = tracer.get_outliers(self.weight)
                    
                    if out_idx is not None and out_idx.numel() > 0:
                        self._set_selected_cols_from_idx(out_idx)
            except Exception:
                pass
        if self.W_int8 is None or self.W_scale is None:
            self.prepare()
        B, T, K = x.shape
        assert K == self.nx
        N = B * T
        x2d = x.reshape(N, K)  # (N, nx)
        
        ######################### ##################
        ######################### ##################
        #######    MUX-Q  ####### ### per-vector ###
        ######################### ##################
        ######################### ##################

        # # ===== 1) outlier column decomposition =====
        # if self.sel_mask.any():
        #     out_idx = self.sel_idx   # (Ks,)
        #     x_main, x_out = self.split_outlier_columns_fp(x2d, out_idx)
        # else:
        #     x_main = x2d
        #     x_out = None
        # # ===== 2) weight scale (per-output-channel) =====
        # s_w = self.W_scale.to(x2d.dtype)       # (nf,)
        # s_w_row = s_w.view(1, self.nf)         # (1, nf)

        # # ===== 3) main path: x_main per-row INT8 quant =====
        # if x_main is not None and x_main.numel() > 0:
        #     x_main_q8, s_main_row = quantize_int8_per_row(x_main)   # (N,K), (N,)
        #     W_main_q8 = self.W_int8                                 # (K,nf)

        #     x_main_deq = x_main_q8.float() * s_main_row.view(-1, 1) # (N,K)
        #     W_main_deq = W_main_q8.float() * s_w_row                # (K,nf)

        #     y_main = torch.matmul(x_main_deq, W_main_deq)           # (N,nf)
        # else:
        #     y_main = torch.zeros((N, self.nf),
        #                         device=x2d.device,
        #                         dtype=torch.float32)
        
        # # ===== 4) outlier path: x_out per-row INT8 quant =====
        # if x_out is not None and x_out.numel() > 0:
        #     x_out_q8, s_out_row = quantize_int8_per_row(x_out)      # (N,Ks), (N,)
        #     W_out_q8 = self.W_int8[out_idx, :]                      # (Ks,nf)

        #     x_out_deq = x_out_q8.float() * s_out_row.view(-1, 1)    # (N,Ks)
        #     W_out_deq = W_out_q8.float() * s_w_row                  # (Ks,nf)

        #     y_out = torch.matmul(x_out_deq, W_out_deq)              # (N,nf)
        # else:
        #     y_out = torch.zeros_like(y_main)



        ######################### ##################
        ######################### ##################
        #######    MUX-Q  ####### ### per-tensor ###
        ######################### ##################
        ######################### ##################


        eps = 1e-12
        ## you can adjust per-tensor quantization bit here

        qmax_a = 2**7 - 1  # 127
        qmax_w = 2**7 - 1
        if self.sel_mask.any():
            out_idx = self.sel_idx   # (Ks,)
            x_main, x_out = self.split_outlier_columns_fp(x2d, out_idx)
        else:
            x_main = x2d
            x_out = None
        


        # ===== 2) weight scale (per-output-channel) =====
        w_amax = self.W_fp.abs().max().clamp_min(eps)
        s_w = w_amax / qmax_w                          # scalar
        W_q8 = torch.round(self.W_fp / s_w).clamp_(-qmax_w-1, qmax_w).to(torch.int8)   # (K, nf)

        # ===== 3) main path: x_main per-tensor INT8 quant =====
        if x_main is not None and x_main.numel() > 0:

            s_main = x_main.abs().max().clamp_min(1e-12) / qmax_a
            x_main_q8 = torch.round(x_main / s_main).clamp_(-qmax_a-1, qmax_a).to(torch.int8)

            x_main_deq = x_main_q8.float() * s_main
            W_main_deq = W_q8.float() * s_w
            y_main = torch.matmul(x_main_deq, W_main_deq)
        else:
            y_main = torch.zeros((N, self.nf), device=x2d.device, dtype=torch.float32)

        
        # ===== 4) outlier path: x_out per-tensor INT8 quant =====
        if x_out is not None and x_out.numel() > 0:
            amax_out = x_out.abs().max().clamp_min(eps)
            s_out = amax_out / qmax_a                          # scalar
            x_out_q8 = torch.round(x_out / s_out).clamp_(-qmax_a-1, qmax_a).to(torch.int8)

            W_out_q8 = W_q8[out_idx, :]                     # (Ks, nf)

            # dequant
            x_out_deq = x_out_q8.float() * s_out           # (N, Ks)
            W_out_deq = W_out_q8.float() * s_w             # (Ks, nf)

            y_out = torch.matmul(x_out_deq, W_out_deq)     # (N, nf)
        else:
            y_out = torch.zeros_like(y_main)

        #########################
        #########################
        #####    LLM.INT8   #####
        #########################
        #########################

        # # ===== 1) outlier column  =====
        # if self.sel_mask.any():
        #     out_idx = self.sel_idx                      # (Ks,)
        #     x_main, x_out = self.split_outlier_columns(x2d, out_idx)
        # else:
        #     out_idx = None
        #     x_main = x2d
        #     x_out  = None

        # # ===== 2) weight scale (per-output-channel) =====
        # s_w = self.W_scale.to(x2d.dtype)               # (nf,)
        # s_w_row = s_w.view(1, self.nf)                 # (1, nf)

        # # ===== 3) main path: x_main per-row INT8 quant, W INT8 =====
        # if x_main is not None and x_main.numel() > 0:
        #     # (N, K), (N,)
        #     x_main_q8, s_main_row = quantize_int8_per_row(x_main)

        #     W_main_q8 = self.W_int8                    # (K, nf)
        #     if self.sel_mask.any():
        #         # sel_mask: (K,)  -> True row(outlier)  0
        #         W_main_q8 = W_main_q8.masked_fill(
        #             self.sel_mask.view(-1, 1), 0
        #         )
            
        #     # dequant (float32 연산)
        #     x_main_deq = x_main_q8.float() * s_main_row.view(-1, 1)  # (N, K)
        #     W_main_deq = W_main_q8.float() * s_w_row                 # (K, nf)

        #     y_main = torch.matmul(x_main_deq, W_main_deq)            # (N, nf)
        # else:
        #     y_main = torch.zeros((N, self.nf),
        #                         device=x2d.device,
        #                         dtype=torch.float132)

        # # ===== 4) outlier path: x_out × W_fp16 =====
        # if x_out is not None and x_out.numel() > 0:

        #     W_out_fp = self.W_fp[out_idx, :].to(torch.float16)                           # (Ks, nf)

        #     # x_out W_out_fp dtype 
        #     x_out_fp = x_out.to(W_out_fp.dtype)                         # (N, Ks)
        #     # high-precision path
        #     y_out = torch.matmul(x_out_fp, W_out_fp)                    # (N, nf)
        # else:
        #     y_out = torch.zeros_like(y_main)



        ######################### ##################
        ######################### ##################
        ##### LLM.int8() ######## per-tensor ###
        ######################### ##################
        ######################### ##################

        # eps = 1e-12

        # ## you can adjust per-tensor quantization bit here
        # qmax_a = 2**7 - 1  # 127
        # qmax_w = 2**7 - 1  # 127
        # # ===== 1) outlier column  =====
        # if self.sel_mask.any():
        #     out_idx = self.sel_idx   # (Ks,)
        #     x_main, x_out = self.split_outlier_columns(x2d, out_idx)
        # else:
        #     x_main = x2d
        #     x_out = None

        # # ===== 2) weight scale (per-output-channel) =====
        # w_amax = self.W_fp.abs().max().clamp_min(eps)
        # s_w = w_amax / qmax_w                          # scalar
        # W_q8 = torch.round(self.W_fp / s_w).clamp_(-qmax_w-1, qmax_w).to(torch.int8)   # (K, nf)

        # # ===== 3) main path: x_main per-tensor INT8 quant =====
        # if x_main is not None and x_main.numel() > 0:
        #     s_main = x_main.abs().max().clamp_min(1e-12) / qmax_a
        #     x_main_q8 = torch.round(x_main / s_main).clamp_(-qmax_a-1, qmax_a).to(torch.int8)

        #     if self.sel_mask.any():
        #         W_q8 = W_q8.masked_fill(
        #             self.sel_mask.view(-1, 1), 0
        #         )

        #     x_main_deq = x_main_q8.float() * s_main
        #     W_main_deq = W_q8.float() * s_w
        #     y_main = torch.matmul(x_main_deq, W_main_deq)
        # else:
        #     y_main = torch.zeros((N, self.nf), device=x2d.device, dtype=torch.float32)

        
        # # ===== 4) outlier path: x_out × W_fp16  =====
        # if x_out is not None and x_out.numel() > 0:

        #     W_out_fp = self.W_fp[out_idx, :].to(torch.float16)                           # (Ks, nf)
        #     x_out_fp = x_out.to(W_out_fp.dtype)                         # (N, Ks)
        #     y_out = torch.matmul(x_out_fp, W_out_fp)                    # (N, nf)
        # else:
        #     y_out = torch.zeros_like(y_main)




        #####################################
        ######### merge : main + out ########
        #####################################
#        y = y_main   + y_out   + self.bias.view(1, self.nf) # alpha = 1 beta = 1
        y = y_main   + y_out*3 + self.bias.view(1, self.nf) # alpha = 2 beta = 2
#        y = y_main   + y_out*7 + self.bias.view(1, self.nf) # alpha = 4 beta = 4
#        y = y_main   + y_out*2 + self.bias.view(1, self.nf) # alpha = 1 beta =2

        y = y.to(torch.float32)
        return y.view(B, T, self.nf)

    def export_mix_params(self):
        return {
            "W_int8": self.W_int8,       # (nx, nf) int8
            "W_scale": self.W_scale,     # (nf,) float
            "selected_cols": self.selected_cols,
        }



        ##############################################################
        ##############################################################
        ##############################################################
        ##############################################################

