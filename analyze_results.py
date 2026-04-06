import pandas as pd
import json
import argparse
from pathlib import Path
import sys

try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    HAS_PLOT = True
except ImportError:
    HAS_PLOT = False
    print("Warning: matplotlib/seaborn not installed. Plotting disabled.")


class ResultAnalyzer:
    def __init__(self, result_dir):
        self.result_dir = Path(result_dir)
        self.csv_file = self.result_dir / "results.csv"
        self.json_file = self.result_dir / "results.json"
        
        if self.csv_file.exists():
            self.df = pd.read_csv(self.csv_file)
        else:
            raise FileNotFoundError(f"Results file not found: {self.csv_file}")
    
    def print_summary(self):
        """결과 요약 출력"""
        print("\n" + "="*80)
        print("RESULTS SUMMARY")
        print("="*80 + "\n")
        
        # 성공한 실험만 필터링
        successful = self.df[self.df['status'] == 'success'].copy()
        
        if len(successful) == 0:
            print("No successful experiments found!")
            return
        
        print(f"Total successful experiments: {len(successful)}\n")
        
        # 모델별 최고 성능
        print("Best PPL by Model:")
        print("-" * 80)
        print(f"{'Model':<30} {'Mode':<12} {'Act_bits':<10} {'PPL':<12}")
        print("-" * 80)
        
        for model in successful['model'].unique():
            model_results = successful[successful['model'] == model].sort_values('ppl')
            best = model_results.iloc[0]
            model_short = model.split('/')[-1]
            print(f"{model_short:<30} {best['quant_mode']:<12} {int(best['act_bits']):<10} {best['ppl']:<12.4f}")
        
        print("\n")
    
    def print_detailed_results(self):
        """상세 결과 출력"""
        print("="*80)
        print("DETAILED RESULTS")
        print("="*80 + "\n")
        
        successful = self.df[self.df['status'] == 'success'].copy()
        
        if len(successful) == 0:
            print("No successful experiments found!")
            return
        
        # 모드별 비교
        print("Results by Quantization Mode:")
        print("-" * 80)
        for mode in successful['quant_mode'].unique():
            mode_results = successful[successful['quant_mode'] == mode].sort_values('ppl')
            print(f"\nMode: {mode}")
            print(f"{'Model':<30} {'Act_bits':<10} {'Weight_bits':<12} {'PPL':<12}")
            print("-" * 60)
            for _, row in mode_results.iterrows():
                model_short = row['model'].split('/')[-1]
                print(f"{model_short:<30} {int(row['act_bits']):<10} {int(row['weight_bits']):<12} {row['ppl']:<12.4f}")
        
        print("\n")
    
    def print_comparison(self):
        """Per-vector vs Per-tensor 비교"""
        print("="*80)
        print("PER-VECTOR vs PER-TENSOR COMPARISON")
        print("="*80 + "\n")
        
        successful = self.df[self.df['status'] == 'success'].copy()
        
        if len(successful) == 0:
            print("No successful experiments found!")
            return
        
        for model in successful['model'].unique():
            model_results = successful[successful['model'] == model]
            model_short = model.split('/')[-1]
            
            print(f"\nModel: {model_short}")
            print("-" * 60)
            print(f"{'Config':<40} {'Per-Vector':<15} {'Per-Tensor':<15} {'Diff':<12}")
            print("-" * 60)
            
            for act_bits in sorted(model_results['act_bits'].unique()):
                for weight_bits in sorted(model_results['weight_bits'].unique()):
                    config = f"Act={int(act_bits)}, Weight={int(weight_bits)}"
                    
                    pv = model_results[(model_results['act_bits'] == act_bits) & 
                                       (model_results['weight_bits'] == weight_bits) &
                                       (model_results['quant_mode'] == 'per-vector')]
                    pt = model_results[(model_results['act_bits'] == act_bits) & 
                                       (model_results['weight_bits'] == weight_bits) &
                                       (model_results['quant_mode'] == 'per-tensor')]
                    
                    pv_ppl = pv['ppl'].values[0] if len(pv) > 0 else None
                    pt_ppl = pt['ppl'].values[0] if len(pt) > 0 else None
                    
                    pv_str = f"{pv_ppl:.4f}" if pv_ppl else "N/A"
                    pt_str = f"{pt_ppl:.4f}" if pt_ppl else "N/A"
                    
                    if pv_ppl and pt_ppl:
                        diff = pt_ppl - pv_ppl
                        diff_str = f"{diff:+.4f}"
                        print(f"{config:<40} {pv_str:<15} {pt_str:<15} {diff_str:<12}")
                    else:
                        print(f"{config:<40} {pv_str:<15} {pt_str:<15} {'N/A':<12}")
        
        print("\n")
    
    def generate_plots(self, output_dir=None):
        """결과 시각화"""
        if not HAS_PLOT:
            print("Plotting disabled (matplotlib not installed)")
            return
        
        if output_dir is None:
            output_dir = self.result_dir
        else:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
        
        successful = self.df[self.df['status'] == 'success'].copy()
        
        if len(successful) == 0:
            print("No successful experiments found!")
            return
        
        # Plot 1: 모델별 성능
        plt.figure(figsize=(12, 6))
        for mode in successful['quant_mode'].unique():
            mode_data = successful[successful['quant_mode'] == mode].sort_values('ppl')
            plt.plot(range(len(mode_data)), mode_data['ppl'].values, marker='o', label=mode)
        plt.xlabel('Experiment Index')
        plt.ylabel('Perplexity')
        plt.title('Model Performance Comparison')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_dir / "performance_comparison.png", dpi=300)
        print(f"✓ Saved: performance_comparison.png")
        plt.close()
        
        # Plot 2: 모드별 박스플롯
        plt.figure(figsize=(10, 6))
        successful['model_short'] = successful['model'].apply(lambda x: x.split('/')[-1])
        data_for_plot = successful[['model_short', 'quant_mode', 'ppl']].copy()
        
        sns.boxplot(data=data_for_plot, x='model_short', y='ppl', hue='quant_mode')
        plt.xlabel('Model')
        plt.ylabel('Perplexity')
        plt.title('Perplexity Distribution by Model and Mode')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(output_dir / "boxplot_by_model.png", dpi=300)
        print(f"✓ Saved: boxplot_by_model.png")
        plt.close()
        
        # Plot 3: Bit별 성능
        plt.figure(figsize=(12, 6))
        for mode in successful['quant_mode'].unique():
            mode_data = successful[successful['quant_mode'] == mode].groupby('act_bits')['ppl'].mean()
            plt.plot(mode_data.index, mode_data.values, marker='o', label=mode, linewidth=2)
        plt.xlabel('Activation Bits')
        plt.ylabel('Average Perplexity')
        plt.title('Perplexity vs Activation Bits')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_dir / "ppl_vs_bits.png", dpi=300)
        print(f"✓ Saved: ppl_vs_bits.png")
        plt.close()
        
        print(f"\nAll plots saved to: {output_dir}\n")


def main():
    parser = argparse.ArgumentParser(description='MUXQ Results Analyzer')
    parser.add_argument('result_dir', type=str,
                        help='Directory containing results (should have results.csv)')
    parser.add_argument('--detailed', action='store_true',
                        help='Print detailed results')
    parser.add_argument('--comparison', action='store_true',
                        help='Print per-vector vs per-tensor comparison')
    parser.add_argument('--plot', action='store_true',
                        help='Generate plots')
    parser.add_argument('--all', action='store_true',
                        help='Print all outputs')
    
    args = parser.parse_args()
    
    try:
        analyzer = ResultAnalyzer(args.result_dir)
        
        if args.all:
            analyzer.print_summary()
            analyzer.print_detailed_results()
            analyzer.print_comparison()
            analyzer.generate_plots()
        else:
            if not args.detailed and not args.comparison and not args.plot:
                analyzer.print_summary()
            
            if args.detailed:
                analyzer.print_detailed_results()
            
            if args.comparison:
                analyzer.print_comparison()
            
            if args.plot:
                analyzer.generate_plots()
    
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
