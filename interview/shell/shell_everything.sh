#!/bin/bash
# Shell Scripting — Everything for ML Engineers
# Run: bash shell_everything.sh

echo "============================================"
echo "SHELL SCRIPTING — ML ENGINEER GUIDE"
echo "============================================"

# ════════════════════════════════════════════
# 1. VARIABLES & STRINGS
# ════════════════════════════════════════════
echo -e "\n1. VARIABLES & STRINGS"

NAME="Keerthi"
AGE=25
PI=3.14

echo "Name: $NAME"                        # Keerthi
echo "Age: $AGE, PI: $PI"                 # Age: 25, PI: 3.14
echo "Length of name: ${#NAME}"           # 7
echo "Upper: ${NAME^^}"                   # KEERTHI
echo "Lower: ${NAME,,}"                   # keerthi
echo "Replace: ${NAME/ee/EE}"             # KEErthi
echo "Substring: ${NAME:0:4}"             # Keer

# Readonly variable
readonly CONFIG_FILE="config.yaml"

# Default value
PORT=${PORT:-8000}        # use PORT env var, or default to 8000
echo "Port: $PORT"

# String concatenation
FULL="$NAME is $AGE years old"
echo "$FULL"

# ════════════════════════════════════════════
# 2. CONDITIONALS
# ════════════════════════════════════════════
echo -e "\n2. CONDITIONALS"

# File checks
if [ -f "requirements.txt" ]; then
    echo "requirements.txt exists"
else
    echo "requirements.txt NOT found"
fi

# Directory check
if [ -d "./data" ]; then
    echo "data/ directory exists"
fi

# Number comparison
SCORE=95
if [ $SCORE -ge 90 ]; then
    echo "Grade: A"
elif [ $SCORE -ge 80 ]; then
    echo "Grade: B"
else
    echo "Grade: C"
fi

# String comparison
ENV="production"
if [[ "$ENV" == "production" ]]; then
    echo "Running in PRODUCTION — be careful!"
fi

# Check command exists
if command -v python3 &>/dev/null; then
    echo "Python3 available: $(python3 --version)"
fi

# Case statement
MODEL_TYPE="classification"
case "$MODEL_TYPE" in
    "classification") echo "Using accuracy + F1 metrics" ;;
    "regression")     echo "Using RMSE + R2 metrics" ;;
    "clustering")     echo "Using silhouette score" ;;
    *)                echo "Unknown model type" ;;
esac

# ════════════════════════════════════════════
# 3. LOOPS
# ════════════════════════════════════════════
echo -e "\n3. LOOPS"

# For loop
for model in "bert" "gpt" "t5" "llama"; do
    echo "  Processing: $model"
done

# Loop with range
for i in {1..5}; do
    echo "  Epoch $i/5"
done

# Loop with step
for i in $(seq 0 10 100); do    # 0 to 100, step 10
    echo -n "$i "
done
echo ""

# While loop
COUNT=0
while [ $COUNT -lt 3 ]; do
    echo "  Count: $COUNT"
    (( COUNT++ ))
done

# Loop over files
for file in *.py 2>/dev/null; do
    if [ -f "$file" ]; then
        lines=$(wc -l < "$file")
        echo "  $file: $lines lines"
    fi
done

# Loop over directories
for dir in data models logs; do
    mkdir -p "$dir"
    echo "  Created: $dir/"
done

# ════════════════════════════════════════════
# 4. FUNCTIONS
# ════════════════════════════════════════════
echo -e "\n4. FUNCTIONS"

# Basic function
greet() {
    local name=$1       # local variable
    local greeting=${2:-"Hello"}   # default arg
    echo "$greeting, $name!"
}

greet "Keerthi"           # Hello, Keerthi!
greet "Alice" "Hi"        # Hi, Alice!

# Logging function
log() {
    local level=$1
    local message=$2
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] [$level] $message"
}

log "INFO"  "Training started"
log "ERROR" "Model failed to converge"
log "WARN"  "GPU memory low"

# Function with return value
get_file_count() {
    local dir=$1
    local ext=${2:-"*"}
    echo $(find "$dir" -name "*.$ext" 2>/dev/null | wc -l)
}

# py_count=$(get_file_count "." "py")
# echo "Python files: $py_count"

# Setup project structure
setup_ml_project() {
    local project=$1
    log "INFO" "Setting up project: $project"
    mkdir -p "$project"/{data/{raw,processed,features},
                        models/{saved,checkpoints},
                        notebooks,src,tests,logs,configs}
    touch "$project/requirements.txt"
    touch "$project/README.md"
    touch "$project/.gitignore"
    cat > "$project/.gitignore" << 'EOF'
__pycache__/
*.pyc
*.pkl
*.pt
venv/
.env
data/raw/
*.log
EOF
    log "INFO" "Project $project created!"
}

# setup_ml_project "my_ml_project"

# ════════════════════════════════════════════
# 5. ARRAYS
# ════════════════════════════════════════════
echo -e "\n5. ARRAYS"

# Declare
models=("RandomForest" "XGBoost" "LightGBM" "BERT")

echo "First: ${models[0]}"          # RandomForest
echo "Last:  ${models[-1]}"         # BERT
echo "All:   ${models[@]}"          # all elements
echo "Count: ${#models[@]}"         # 4

# Add element
models+=("GPT4")

# Loop array
for model in "${models[@]}"; do
    echo "  - $model"
done

# Associative array (dict)
declare -A metrics
metrics["accuracy"]=0.95
metrics["f1"]=0.92
metrics["auc"]=0.98

for key in "${!metrics[@]}"; do
    echo "  $key: ${metrics[$key]}"
done

# ════════════════════════════════════════════
# 6. INPUT / OUTPUT
# ════════════════════════════════════════════
echo -e "\n6. I/O"

# Read from stdin
# read -p "Enter model name: " model_name
# echo "Training: $model_name"

# Redirect output
# echo "Training log" >> training.log    # append
# echo "Fresh log"   >  training.log     # overwrite
# python train.py > output.log 2>&1      # stdout + stderr to file
# python train.py 2>/dev/null            # suppress errors

# Pipe
# cat data.csv | head -5
# python train.py | tee training.log     # print AND save

# Here document
cat << 'EOF' > /tmp/config.yaml
model:
  name: RandomForest
  n_estimators: 200
  max_depth: 5
training:
  epochs: 50
  batch_size: 32
EOF
echo "Config written to /tmp/config.yaml"

# ════════════════════════════════════════════
# 7. ARGUMENT PARSING
# ════════════════════════════════════════════
echo -e "\n7. ARGUMENT PARSING"

# $0 = script name, $1 = first arg, $# = number of args
# $@ = all args, $? = exit code of last command

parse_args() {
    local EPOCHS=50
    local LR=0.001
    local GPU=0
    local CONFIG=""

    while [[ "$#" -gt 0 ]]; do
        case $1 in
            --epochs)  EPOCHS="$2";  shift ;;
            --lr)      LR="$2";      shift ;;
            --gpu)     GPU="$2";     shift ;;
            --config)  CONFIG="$2";  shift ;;
            --help|-h)
                echo "Usage: $0 [--epochs N] [--lr RATE] [--gpu ID]"
                exit 0 ;;
            *) echo "Unknown: $1"; exit 1 ;;
        esac
        shift
    done

    echo "Config: epochs=$EPOCHS, lr=$LR, gpu=$GPU"
    # CUDA_VISIBLE_DEVICES=$GPU python train.py --epochs $EPOCHS --lr $LR
}

parse_args --epochs 100 --lr 0.001 --gpu 0

# ════════════════════════════════════════════
# 8. ML-SPECIFIC SCRIPTS
# ════════════════════════════════════════════
echo -e "\n8. ML SCRIPTS"

# Check GPU
check_gpu() {
    if command -v nvidia-smi &>/dev/null; then
        echo "GPU available:"
        nvidia-smi --query-gpu=name,memory.total,memory.free \
                   --format=csv,noheader
    else
        echo "No GPU found — using CPU"
    fi
}
check_gpu

# Setup Python environment
setup_env() {
    log "INFO" "Setting up virtual environment"
    python3 -m venv venv
    source venv/bin/activate
    pip install --upgrade pip -q
    pip install -r requirements.txt -q
    log "INFO" "Environment ready!"
}

# Run training with retry
run_with_retry() {
    local command=$1
    local max_retries=${2:-3}
    local attempt=0

    while [ $attempt -lt $max_retries ]; do
        log "INFO" "Attempt $((attempt+1))/$max_retries: $command"
        if eval "$command"; then
            log "INFO" "Success!"
            return 0
        fi
        attempt=$((attempt+1))
        [ $attempt -lt $max_retries ] && sleep 5
    done
    log "ERROR" "Failed after $max_retries attempts"
    return 1
}

# run_with_retry "python train.py" 3

# Monitor training process
monitor_process() {
    local pid=$1
    while kill -0 "$pid" 2>/dev/null; do
        CPU=$(ps -p $pid -o %cpu --no-headers 2>/dev/null || echo "N/A")
        MEM=$(ps -p $pid -o %mem --no-headers 2>/dev/null || echo "N/A")
        log "INFO" "PID $pid — CPU: $CPU%, MEM: $MEM%"
        sleep 30
    done
    log "INFO" "Process $pid finished"
}

# python train.py &
# monitor_process $! &

# Data download
download_dataset() {
    local url=$1
    local output=$2
    log "INFO" "Downloading: $url"
    wget -q --show-progress "$url" -O "$output"
    log "INFO" "Downloaded: $output ($(du -sh "$output" | cut -f1))"
}

# Clean old logs and checkpoints
cleanup() {
    log "INFO" "Cleaning old files..."
    find ./logs -name "*.log" -mtime +7 -exec gzip {} \;       # compress old logs
    find ./models/checkpoints -name "*.pt" -mtime +3 -delete   # delete old checkpoints
    log "INFO" "Cleanup done!"
}

# ════════════════════════════════════════════
# 9. USEFUL ONE-LINERS
# ════════════════════════════════════════════
echo -e "\n9. USEFUL ONE-LINERS"

echo "Show Python version:"
python3 --version 2>/dev/null || echo "Python3 not found"

echo "Count lines of Python code:"
find . -name "*.py" -not -path "*/venv/*" | xargs wc -l 2>/dev/null | tail -1

echo "Show disk usage of current directory:"
du -sh .

echo "Show top 5 largest files:"
find . -type f -not -path "*/venv/*" -exec du -sh {} + 2>/dev/null | sort -rh | head -5

echo "Kill process on port 8000:"
# lsof -ti:8000 | xargs kill -9 2>/dev/null || echo "Nothing on port 8000"

echo "Watch GPU every 2 seconds:"
# watch -n 2 nvidia-smi

echo "SSH + run training on remote:"
# ssh user@server "cd /project && source venv/bin/activate && python train.py"

echo "Copy files to remote:"
# rsync -avz --progress ./models/ user@server:/remote/models/

echo ""
echo "============================================"
echo "All done! ✓"
echo "============================================"
