import torch
from tqdm import tqdm
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
import argparse
import torch.nn as nn


# Import from local modules
from out_utils2 import (
    OutlierTracer, MixedInputInt8Conv1D, 
    activation_outlier_hook, find_outlier_dims
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default='openai-community/gpt2-large', 
                        help="Model path or HuggingFace model ID")
    parser.add_argument("--zscore", type=float, default=3.0, 
                        help="Z-score threshold for outlier detection")
    parser.add_argument("--out_mag", type=float, default=5.0, 
                        help="Magnitude threshold for outlier detection")
    parser.add_argument("--split_exponent", type=int, default=2, 
                        help="Split exponent for outlier decomposition")
    parser.add_argument("--quant_method", type=str, default='muxq', 
                        choices=['muxq', 'llm-int8', 'naive'], 
                        help="Quantization method")
    parser.add_argument("--quant_mode", type=str, default='per-tensor', 
                        choices=['per-vector', 'per-tensor'], 
                        help="Quantization mode")
    parser.add_argument("--act_bits", type=int, default=8, 
                        help="Activation quantization bits")
    parser.add_argument("--weight_bits", type=int, default=8, 
                        help="Weight quantization bits")
    parser.add_argument("--device", type=str, default="cuda", 
                        help="Device to use (cuda or cpu)")
    return parser.parse_args()


def replace_conv1d_with_mixed_int8(model, device):
    """
    Replace Conv1D layers in attention and MLP with MixedInputInt8Conv1D
    Target layers: c_attn, c_proj (attention), c_fc, c_proj (mlp)
    """
    replaced_layers = []
    
    for name, module in model.named_modules():
        # Check for attention layers (c_attn, c_proj in self_attn)
        if hasattr(module, 'c_attn') and isinstance(module.c_attn, nn.Module):
            # Get dimensions
            if hasattr(module.c_attn, 'nf') and hasattr(module.c_attn, 'nx'):
                nf = module.c_attn.nf
                nx = module.c_attn.nx
                mixed_layer = MixedInputInt8Conv1D(nf, nx).to(device)
                # Copy weights
                with torch.no_grad():
                    mixed_layer.weight.copy_(module.c_attn.weight)
                    if module.c_attn.bias is not None:
                        mixed_layer.bias.copy_(module.c_attn.bias)
                module.c_attn = mixed_layer
                replaced_layers.append(f"{name}.c_attn")
                print(f"✓ Replaced {name}.c_attn with MixedInputInt8Conv1D")
        
        if hasattr(module, 'c_proj') and isinstance(module.c_proj, nn.Module):
            if hasattr(module.c_proj, 'nf') and hasattr(module.c_proj, 'nx'):
                nf = module.c_proj.nf
                nx = module.c_proj.nx
                mixed_layer = MixedInputInt8Conv1D(nf, nx).to(device)
                # Copy weights
                with torch.no_grad():
                    mixed_layer.weight.copy_(module.c_proj.weight)
                    if module.c_proj.bias is not None:
                        mixed_layer.bias.copy_(module.c_proj.bias)
                module.c_proj = mixed_layer
                replaced_layers.append(f"{name}.c_proj")
                print(f"✓ Replaced {name}.c_proj with MixedInputInt8Conv1D")
        
        # Check for MLP layers (c_fc, c_proj in mlp)
        if hasattr(module, 'c_fc') and isinstance(module.c_fc, nn.Module):
            if hasattr(module.c_fc, 'nf') and hasattr(module.c_fc, 'nx'):
                nf = module.c_fc.nf
                nx = module.c_fc.nx
                mixed_layer = MixedInputInt8Conv1D(nf, nx).to(device)
                # Copy weights
                with torch.no_grad():
                    mixed_layer.weight.copy_(module.c_fc.weight)
                    if module.c_fc.bias is not None:
                        mixed_layer.bias.copy_(module.c_fc.bias)
                module.c_fc = mixed_layer
                replaced_layers.append(f"{name}.c_fc")
                print(f"✓ Replaced {name}.c_fc with MixedInputInt8Conv1D")
    
    return replaced_layers


def replace_conv1d_layers_via_iteration(model, device):
    """
    Iterate through model and replace Conv1D layers (c_attn, c_proj, c_fc) with MixedInputInt8Conv1D
    
    Note: transformers.Conv1D weight shape is (in_features, out_features) - transpose of standard Conv1D
    MixedInputInt8Conv1D expects (out_features, in_features)
    """
    target_names = ['c_attn', 'c_proj', 'c_fc']
    replaced_count = 0
    
    def replace_in_module(mod, parent_name=""):
        nonlocal replaced_count
        
        for child_name, child_module in mod.named_children():
            full_name = f"{parent_name}.{child_name}" if parent_name else child_name
            
            # Check if this is one of our target layers
            if any(target in child_name for target in target_names):
                # Check if it has weight attribute (Conv1D-like layer)
                if hasattr(child_module, 'weight') and child_module.weight is not None:
                    try:
                        # transformers.Conv1D weight shape: (in_features, out_features)
                        # We need to transpose to get (out_features, in_features) for MixedInputInt8Conv1D
                        orig_weight = child_module.weight
                        nx, nf = orig_weight.shape[0], orig_weight.shape[1]  # swap for correct interpretation
                        
                        # Create MixedInputInt8Conv1D with correct dimensions
                        # MixedInputInt8Conv1D expects (nf, nx) = (out_features, in_features)
                        mixed_layer = MixedInputInt8Conv1D(nf, nx).to(device)
                        
                        # Copy weights (transpose needed) and bias
                        with torch.no_grad():
                            # Original shape: (nx, nf), Target shape: (nf, nx)
                            mixed_layer.weight.copy_(orig_weight.t())
                            if hasattr(child_module, 'bias') and child_module.bias is not None:
                                mixed_layer.bias.copy_(child_module.bias)
                        
                        # Prepare quantization (quantize weights)
                        mixed_layer.prepare(weight_bits=8)
                        
                        # Replace the layer in parent
                        setattr(mod, child_name, mixed_layer)
                        replaced_count += 1
                        print(f"✓ Replaced {full_name}: original weight {orig_weight.shape} -> {mixed_layer.weight.shape}")
                    
                    except Exception as e:
                        print(f"✗ Failed to replace {full_name}: {str(e)}")
            
            # Recursively process child modules
            replace_in_module(child_module, full_name)
    
    replace_in_module(model)
    return replaced_count


def evaluate_gpt2_with_muxq(model, tokenizer, device, max_length):
    """
    Evaluate GPT2 model on WikiText-2 test set
    """
    print("\n" + "="*60)
    print("Loading WikiText-2 dataset...")
    print("="*60)
    
    test = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    encodings = tokenizer("\n\n".join(test["text"]), return_tensors="pt")
    
    stride = 512
    seq_len = encodings.input_ids.size(1)
    
    print(f"Total sequence length: {seq_len}")
    print(f"Stride: {stride}, Max length: {max_length}")
    print("\n" + "="*60)
    print("Starting evaluation...")
    print("="*60 + "\n")
    
    nlls = []
    prev_end_loc = 0
    
    for begin_loc in tqdm(range(0, seq_len, stride), desc="Evaluating"):
        end_loc = min(begin_loc + max_length, seq_len)
        trg_len = end_loc - prev_end_loc
        
        input_ids = encodings.input_ids[:, begin_loc:end_loc].to(device)
        target_ids = input_ids.clone()
        target_ids[:, :-trg_len] = -100
        
        with torch.no_grad():
            outputs = model(input_ids, labels=target_ids)
            neg_log_likelihood = outputs.loss
        
        nlls.append(neg_log_likelihood)
        prev_end_loc = end_loc
        
        if end_loc == seq_len:
            break
    
    ppl = torch.exp(torch.stack(nlls).mean())
    
    print("\n" + "="*60)
    print(f"Perplexity (PPL): {ppl.item():.4f}")
    print("="*60)
    
    return ppl.item()


def main():
    args = parse_args()
    device = args.device
    
    print("\n" + "="*60)
    print("GPT2 Evaluation with MUXQ Quantization")
    print("="*60 + "\n")
    
    # Load model and tokenizer
    print(f"Loading model from: {args.model_path}")
    model = AutoModelForCausalLM.from_pretrained(args.model_path).to(device)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    max_length = model.config.n_positions
    
    print(f"Model loaded successfully!")
    print(f"Model type: {type(model).__name__}")
    print(f"Max sequence length: {max_length}\n")
    
    # Step 1: Replace Conv1D layers with MixedInputInt8Conv1D
    print("="*60)
    print("Step 1: Replacing Conv1D layers with MixedInputInt8Conv1D")
    print("="*60 + "\n")
    
    replaced_count = replace_conv1d_layers_via_iteration(model, device)
    print(f"\nTotal layers replaced: {replaced_count}\n")
    
    # Step 2: Initialize OutlierTracer
    print("="*60)
    print("Step 2: Initializing OutlierTracer")
    print("="*60 + "\n")
    
    tracer = OutlierTracer.get_instance()
    tracer.initialize(
        model,
        zscore=args.zscore,
        out_mag=args.out_mag,
        split_exponent=args.split_exponent,
        quant_method=args.quant_method,
        quant_mode=args.quant_mode,
        act_bits=args.act_bits,
        weight_bits=args.weight_bits
    )
    
    print(f"✓ OutlierTracer initialized")
    print(f"  - Z-score threshold: {args.zscore}")
    print(f"  - Magnitude threshold: {args.out_mag}")
    print(f"  - Split exponent: {args.split_exponent}")
    print(f"  - Quant method: {args.quant_method}")
    print(f"  - Quant mode: {args.quant_mode}")
    print(f"  - Act bits: {args.act_bits}")
    print(f"  - Weight bits: {args.weight_bits}\n")
    
    # Step 3: Evaluate model
    ppl = evaluate_gpt2_with_muxq(model, tokenizer, device, max_length)
    
    print("\n" + "="*60)
    print("Evaluation Complete!")
    print("="*60)
    print(f"Final Perplexity: {ppl:.4f}")
    
    return ppl


if __name__ == "__main__":
    main()
