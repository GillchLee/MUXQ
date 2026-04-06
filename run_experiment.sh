#!/bin/bash

# MUXQ Experiment를 쉽게 실행하기 위한 스크립트

set -e

# 색상 정의
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

print_header() {
    echo -e "\n${BLUE}========================================${NC}"
    echo -e "${BLUE}$1${NC}"
    echo -e "${BLUE}========================================${NC}\n"
}

print_success() {
    echo -e "${GREEN}✓ $1${NC}"
}

print_error() {
    echo -e "${RED}✗ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}! $1${NC}"
}

# 1. 환경 확인
print_header "Checking Environment"

# Python 확인
if ! command -v python3 &> /dev/null; then
    print_error "Python3 not found"
    exit 1
fi
print_success "Python3 found: $(python3 --version)"

# PyTorch 확인
python3 -c "import torch; print(f'PyTorch version: {torch.__version__}')" 2>/dev/null && print_success "PyTorch installed" || {
    print_error "PyTorch not installed"
    exit 1
}

# CUDA 확인
CUDA_AVAILABLE=$(python3 -c "import torch; print(torch.cuda.is_available())")
if [ "$CUDA_AVAILABLE" = "True" ]; then
    CUDA_VERSION=$(python3 -c "import torch; print(torch.version.cuda)")
    print_success "CUDA available (version: $CUDA_VERSION)"
else
    print_warning "CUDA not available (will use CPU)"
fi

# 필요한 모듈 확인
python3 -c "import transformers, datasets" 2>/dev/null && print_success "Required packages installed" || {
    print_error "Missing packages. Install with: pip install transformers datasets"
    exit 1
}

# 2. 실험 옵션 선택
print_header "Experiment Configuration"

echo "Select experiment type:"
echo "1) Quick Test (gpt2-small, 8-bit only)"
echo "2) Standard Test (gpt2-small/medium/large, 4-bit and 8-bit)"
echo "3) Full Comparison (all bit widths and modes)"
echo "4) Custom Configuration"
read -p "Enter choice (1-4): " choice

case $choice in
    1)
        print_header "Running Quick Test"
        python3 experiment_runner.py \
            --models openai-community/gpt2-small \
            --act_bits 8 \
            --weight_bits 8 \
            --modes per-vector per-tensor \
            --device cuda
        ;;
    2)
        print_header "Running Standard Test (4-8 bit sweep)"
        python3 experiment_runner.py \
            --models openai-community/gpt2-small \
                     openai-community/gpt2-medium \
                     openai-community/gpt2-large \
            --act_bits 4 5 6 7 8 \
            --weight_bits 4 5 6 7 8 \
            --modes per-vector per-tensor \
            --device cuda
        ;;
    3)
        print_header "Running Full Comparison (2-8 bit sweep)"
        python3 experiment_runner.py \
            --models openai-community/gpt2-small \
                     openai-community/gpt2-medium \
                     openai-community/gpt2-large \
            --act_bits 2 3 4 5 6 7 8 \
            --weight_bits 2 3 4 5 6 7 8 \
            --modes per-vector per-tensor \
            --device cuda
        ;;
    4)
        print_header "Custom Configuration"
        
        read -p "Enter models (space-separated, or press Enter for default): " models
        if [ -z "$models" ]; then
            models="openai-community/gpt2-small openai-community/gpt2-medium openai-community/gpt2-large"
        fi
        
        read -p "Enter activation bits (space-separated, default: 4 8): " act_bits
        if [ -z "$act_bits" ]; then
            act_bits="4 8"
        fi
        
        read -p "Enter weight bits (space-separated, default: 4 8): " weight_bits
        if [ -z "$weight_bits" ]; then
            weight_bits="4 8"
        fi
        
        read -p "Enter quantization modes (space-separated, default: per-vector per-tensor): " modes
        if [ -z "$modes" ]; then
            modes="per-vector per-tensor"
        fi
        
        read -p "Enter Z-score threshold (default: 3.0): " zscore
        if [ -z "$zscore" ]; then
            zscore="3.0"
        fi
        
        read -p "Enter magnitude threshold (default: 5.0): " out_mag
        if [ -z "$out_mag" ]; then
            out_mag="5.0"
        fi
        
        print_header "Running Custom Experiment"
        python3 experiment_runner.py \
            --models $models \
            --act_bits $act_bits \
            --weight_bits $weight_bits \
            --modes $modes \
            --zscore $zscore \
            --out_mag $out_mag \
            --device cuda
        ;;
    *)
        print_error "Invalid choice"
        exit 1
        ;;
esac

# 3. 결과 분석 옵션
print_header "Results Analysis"

# 최신 결과 폴더 찾기
LATEST_RESULT=$(ls -t results/ 2>/dev/null | head -1)

if [ -z "$LATEST_RESULT" ]; then
    print_error "No results found"
    exit 1
fi

RESULT_DIR="results/$LATEST_RESULT"
print_success "Found results in: $RESULT_DIR"

echo ""
echo "Select analysis:"
echo "1) Print Summary"
echo "2) Print Detailed Results"
echo "3) Print Per-Vector vs Per-Tensor Comparison"
echo "4) Generate Plots"
echo "5) Full Analysis (all of above)"
read -p "Enter choice (1-5): " analysis_choice

case $analysis_choice in
    1)
        python3 analyze_results.py "$RESULT_DIR"
        ;;
    2)
        python3 analyze_results.py "$RESULT_DIR" --detailed
        ;;
    3)
        python3 analyze_results.py "$RESULT_DIR" --comparison
        ;;
    4)
        python3 analyze_results.py "$RESULT_DIR" --plot
        ;;
    5)
        python3 analyze_results.py "$RESULT_DIR" --all
        ;;
    *)
        print_error "Invalid choice"
        ;;
esac

# 4. 완료 메시지
print_header "Done!"
print_success "Results saved in: $RESULT_DIR"
print_success "Log file: $RESULT_DIR/experiment.log"
print_success "CSV results: $RESULT_DIR/results.csv"
print_success "JSON results: $RESULT_DIR/results.json"

echo ""
echo "You can analyze results anytime with:"
echo "  python3 analyze_results.py $RESULT_DIR --all"
echo ""
