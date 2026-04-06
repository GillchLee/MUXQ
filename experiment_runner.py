import torch
import argparse
import json
import csv
import os
from datetime import datetime
from pathlib import Path
from tqdm import tqdm
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
import sys

from out_utils2 import OutlierTracer, MixedInputInt8Conv1D


class ExperimentRunner:
    def __init__(self, output_dir="./results"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 타임스탬프로 실험 폴더 생성
        self.exp_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.exp_dir = self.output_dir / self.exp_timestamp
        self.exp_dir.mkdir(parents=True, exist_ok=True)
        
        self.results = []
        self.log_file = self.exp_dir / "experiment.log"
        self.csv_file = self.exp_dir / "results.csv"
        self.json_file = self.exp_dir / "results.json"
        
    def log(self, message):
        """Print과 동시에 로그 파일에 저장"""
        print(message)
        with open(self.log_file, 'a') as f:
            f.write(message + '\n')
    
    def replace_conv1d_layers(self, model, device):
        """Conv1D 레이어를 MixedInputInt8Conv1D로 교체"""
        target_names = ['c_attn', 'c_proj', 'c_fc']
        replaced_count = 0
        
        def replace_in_module(mod, parent_name=""):
            nonlocal replaced_count
            
            for child_name, child_module in mod.named_children():
                full_name = f"{parent_name}.{child_name}" if parent_name else child_name
                
                if any(target in child_name for target in target_names):
                    if hasattr(child_module, 'weight') and child_module.weight is not None:
                        try:
                            orig_weight = child_module.weight
                            nx, nf = orig_weight.shape[0], orig_weight.shape[1]
                            
                            mixed_layer = MixedInputInt8Conv1D(nf, nx).to(device)
                            
                            with torch.no_grad():
                                mixed_layer.weight.copy_(orig_weight.t())
                                if hasattr(child_module, 'bias') and child_module.bias is not None:
                                    mixed_layer.bias.copy_(child_module.bias)
                            
                            mixed_layer.prepare(weight_bits=8)
                            setattr(mod, child_name, mixed_layer)
                            replaced_count += 1
                        except Exception as e:
                            self.log(f"  Warning: Failed to replace {full_name}: {str(e)}")
                
                replace_in_module(child_module, full_name)
        
        replace_in_module(model)
        return replaced_count
    
    def evaluate_model(self, model, tokenizer, device, max_length):
        """WikiText-2에서 모델 평가"""
        test = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
        encodings = tokenizer("\n\n".join(test["text"]), return_tensors="pt")
        
        stride = 512
        seq_len = encodings.input_ids.size(1)
        
        nlls = []
        prev_end_loc = 0
        
        for begin_loc in tqdm(range(0, seq_len, stride), desc="Evaluating", leave=False):
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
        return ppl.item()
    
    def run_experiment(self, model_name, act_bits, weight_bits, quant_mode, 
                       quant_method, device, zscore=3.0, out_mag=5.0):
        """단일 실험 실행"""
        self.log(f"\n{'='*70}")
        self.log(f"Experiment: Model={model_name}, Act_bits={act_bits}, Weight_bits={weight_bits}")
        self.log(f"            Mode={quant_mode}, Method={quant_method}")
        self.log(f"{'='*70}")
        
        try:
            # 모델 로드
            self.log(f"[1/4] Loading model {model_name}...")
            model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            max_length = model.config.n_positions
            self.log(f"      ✓ Model loaded (max_length={max_length})")
            
            # Conv1D 교체
            self.log(f"[2/4] Replacing Conv1D layers...")
            replaced_count = self.replace_conv1d_layers(model, device)
            self.log(f"      ✓ {replaced_count} layers replaced")
            
            # OutlierTracer 초기화
            self.log(f"[3/4] Initializing OutlierTracer...")
            tracer = OutlierTracer.get_instance()
            tracer.initialize(
                model,
                zscore=zscore,
                out_mag=out_mag,
                split_exponent=2,
                quant_method=quant_method,
                quant_mode=quant_mode,
                act_bits=act_bits,
                weight_bits=weight_bits
            )
            self.log(f"      ✓ OutlierTracer initialized")
            
            # 평가
            self.log(f"[4/4] Evaluating model...")
            ppl = self.evaluate_model(model, tokenizer, device, max_length)
            self.log(f"      ✓ Perplexity: {ppl:.4f}")
            
            # 결과 저장
            result = {
                'model': model_name,
                'act_bits': act_bits,
                'weight_bits': weight_bits,
                'quant_mode': quant_mode,
                'quant_method': quant_method,
                'zscore': zscore,
                'out_mag': out_mag,
                'ppl': ppl,
                'status': 'success'
            }
            
            self.results.append(result)
            return result
        
        except Exception as e:
            self.log(f"      ✗ Error: {str(e)}")
            result = {
                'model': model_name,
                'act_bits': act_bits,
                'weight_bits': weight_bits,
                'quant_mode': quant_mode,
                'quant_method': quant_method,
                'zscore': zscore,
                'out_mag': out_mag,
                'ppl': None,
                'status': f'failed: {str(e)}'
            }
            self.results.append(result)
            return result
    
    def save_results(self):
        """결과를 CSV와 JSON으로 저장"""
        self.log(f"\n{'='*70}")
        self.log("Saving results...")
        self.log(f"{'='*70}")
        
        # CSV 저장
        if self.results:
            keys = self.results[0].keys()
            with open(self.csv_file, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=keys)
                writer.writeheader()
                writer.writerows(self.results)
            self.log(f"✓ CSV saved: {self.csv_file}")
        
        # JSON 저장
        with open(self.json_file, 'w') as f:
            json.dump(self.results, f, indent=2)
        self.log(f"✓ JSON saved: {self.json_file}")
        
        # Log 파일 경로 출력
        self.log(f"✓ Log saved: {self.log_file}")
        self.log(f"\nExperiment directory: {self.exp_dir}")
    
    def print_summary(self):
        """결과 요약 출력"""
        self.log(f"\n{'='*70}")
        self.log("EXPERIMENT SUMMARY")
        self.log(f"{'='*70}\n")
        
        successful = [r for r in self.results if r['status'] == 'success']
        failed = [r for r in self.results if r['status'] != 'success']
        
        self.log(f"Total experiments: {len(self.results)}")
        self.log(f"Successful: {len(successful)}")
        self.log(f"Failed: {len(failed)}\n")
        
        if successful:
            self.log("Results (sorted by PPL):")
            self.log(f"{'Model':<25} {'Mode':<12} {'Act_bits':<10} {'PPL':<12}")
            self.log("-" * 60)
            
            sorted_results = sorted(successful, key=lambda x: x['ppl'])
            for r in sorted_results:
                model_short = r['model'].split('/')[-1]
                self.log(f"{model_short:<25} {r['quant_mode']:<12} {r['act_bits']:<10} {r['ppl']:<12.4f}")
        
        if failed:
            self.log(f"\nFailed experiments:")
            for r in failed:
                self.log(f"  - {r['model']} ({r['status']})")


def main():
    parser = argparse.ArgumentParser(description='MUXQ Experiment Runner')
    parser.add_argument('--models', nargs='+', 
                        default=['openai-community/gpt2-small', 
                                'openai-community/gpt2-medium',
                                'openai-community/gpt2-large'],
                        help='Model names to evaluate')
    parser.add_argument('--act_bits', nargs='+', type=int, default=[4, 5, 6, 7, 8],
                        help='Activation bits to test')
    parser.add_argument('--weight_bits', nargs='+', type=int, default=[4, 5, 6, 7, 8],
                        help='Weight bits to test')
    parser.add_argument('--modes', nargs='+', 
                        default=['per-vector', 'per-tensor'],
                        choices=['per-vector', 'per-tensor'],
                        help='Quantization modes')
    parser.add_argument('--method', type=str, default='muxq',
                        choices=['muxq', 'llm-int8', 'naive'],
                        help='Quantization method')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device: cuda or cpu')
    parser.add_argument('--zscore', type=float, default=3.0,
                        help='Z-score threshold')
    parser.add_argument('--out_mag', type=float, default=5.0,
                        help='Magnitude threshold')
    parser.add_argument('--output_dir', type=str, default='./results',
                        help='Output directory for results')
    
    args = parser.parse_args()
    
    # 실험 런너 초기화
    runner = ExperimentRunner(output_dir=args.output_dir)
    
    runner.log(f"{'='*70}")
    runner.log("MUXQ Experiment Runner")
    runner.log(f"{'='*70}")
    runner.log(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    runner.log(f"\nExperiment Configuration:")
    runner.log(f"  Models: {args.models}")
    runner.log(f"  Act bits: {args.act_bits}")
    runner.log(f"  Weight bits: {args.weight_bits}")
    runner.log(f"  Modes: {args.modes}")
    runner.log(f"  Method: {args.method}")
    runner.log(f"  Device: {args.device}")
    runner.log(f"  Z-score threshold: {args.zscore}")
    runner.log(f"  Magnitude threshold: {args.out_mag}\n")
    
    total_experiments = len(args.models) * len(args.act_bits) * len(args.weight_bits) * len(args.modes)
    runner.log(f"Total experiments to run: {total_experiments}\n")
    
    # 실험 실행
    exp_count = 0
    for model_name in args.models:
        for act_bits in args.act_bits:
            for weight_bits in args.weight_bits:
                for mode in args.modes:
                    exp_count += 1
                    runner.log(f"\n[{exp_count}/{total_experiments}]")
                    runner.run_experiment(
                        model_name=model_name,
                        act_bits=act_bits,
                        weight_bits=weight_bits,
                        quant_mode=mode,
                        quant_method=args.method,
                        device=args.device,
                        zscore=args.zscore,
                        out_mag=args.out_mag
                    )
    
    # 결과 저장
    runner.save_results()
    runner.print_summary()
    
    runner.log(f"\nEnd time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    runner.log(f"{'='*70}")


if __name__ == "__main__":
    main()
