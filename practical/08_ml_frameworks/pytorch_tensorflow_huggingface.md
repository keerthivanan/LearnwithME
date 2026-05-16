# 08 — ML Frameworks: PyTorch, TensorFlow & Hugging Face

> The JD specifically mentions PyTorch, TensorFlow, and Hugging Face. Know each well enough to code in them.

---

## 1. PyTorch

### Why PyTorch?
- Dynamic computation graph (define by run) — easier to debug
- Pythonic, feels natural
- **Dominant in research** and increasingly in production
- Used by: Meta, most AI research labs, Hugging Face

### Core Concepts

**Tensor** — The basic data structure (like NumPy arrays but with GPU support)
```python
import torch

# Create tensors
x = torch.tensor([1.0, 2.0, 3.0])
y = torch.zeros(3, 4)
z = torch.randn(2, 3)       # random normal

# GPU
device = "cuda" if torch.cuda.is_available() else "cpu"
x = x.to(device)
```

**Autograd** — Automatic differentiation
```python
x = torch.tensor([2.0], requires_grad=True)
y = x ** 2 + 3 * x        # y = x² + 3x
y.backward()               # computes dy/dx
print(x.grad)              # tensor([7.]) → dy/dx at x=2 is 2*2+3=7
```

### Building a Neural Network
```python
import torch.nn as nn

class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(768, 256)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(256, 2)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

model = SimpleModel().to(device)
```

### Training Loop
```python
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
criterion = nn.CrossEntropyLoss()

for epoch in range(num_epochs):
    for batch in dataloader:
        inputs, labels = batch
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()           # reset gradients
        outputs = model(inputs)         # forward pass
        loss = criterion(outputs, labels)
        loss.backward()                 # compute gradients
        optimizer.step()               # update weights
```

### Key PyTorch APIs
| Module | Purpose |
|--------|---------|
| `torch.nn` | Neural network layers |
| `torch.optim` | Optimizers (Adam, AdamW, SGD) |
| `torch.utils.data` | Dataset and DataLoader |
| `torch.cuda` | GPU operations |
| `torch.amp` | Mixed precision training |

### Mixed Precision Training (FP16/BF16)
Train in half-precision for speed + memory savings:
```python
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()

with autocast():
    outputs = model(inputs)
    loss = criterion(outputs, labels)

scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()
```
BF16 is preferred for LLMs (more stable than FP16).

---

## 2. TensorFlow / Keras

### Overview
- Developed by Google
- **Static computation graph** (TF1) → **eager execution** (TF2, dynamic like PyTorch)
- Better for production deployment (TensorFlow Serving, TFLite)
- Keras: high-level API, built into TF2

### Building a Model with Keras
```python
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

model = keras.Sequential([
    layers.Dense(256, activation='relu', input_shape=(768,)),
    layers.Dropout(0.2),
    layers.Dense(2, activation='softmax')
])

model.compile(
    optimizer=keras.optimizers.Adam(lr=1e-4),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

model.fit(train_dataset, epochs=3, validation_data=val_dataset)
```

### Custom Training Loop
```python
optimizer = keras.optimizers.Adam(1e-4)
loss_fn = keras.losses.SparseCategoricalCrossentropy()

@tf.function
def train_step(x, y):
    with tf.GradientTape() as tape:
        preds = model(x, training=True)
        loss = loss_fn(y, preds)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    return loss
```

### TensorFlow vs PyTorch
| Feature | PyTorch | TensorFlow/Keras |
|---------|---------|-----------------|
| Computation graph | Dynamic | Both (TF2 = dynamic by default) |
| Research use | Dominant | Less common now |
| Production | TorchServe, ONNX | TF Serving, TFLite |
| Ease of use | More Pythonic | Keras is very simple |
| LLM ecosystem | Hugging Face native | Less integrated |

---

## 3. Hugging Face Transformers (Most Important)

### What is it?
The most widely used library for working with LLMs. Provides pre-trained models, tokenizers, and training utilities.

### Core Components
```
transformers  → Pre-trained models and tokenizers
datasets      → Load and process NLP datasets
peft          → LoRA, adapters, PEFT methods
trl           → SFT, RLHF, DPO training
accelerate    → Multi-GPU, distributed training
evaluate      → Metrics (BLEU, ROUGE, accuracy)
```

### Loading a Model and Tokenizer
```python
from transformers import AutoTokenizer, AutoModelForCausalLM

model_id = "meta-llama/Llama-3.1-8B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    device_map="auto"          # auto-distribute across GPUs
)
```

### Auto Classes
| Class | Task |
|-------|-----|
| `AutoModel` | Raw transformer (embeddings) |
| `AutoModelForCausalLM` | Text generation (GPT) |
| `AutoModelForMaskedLM` | Masked LM (BERT) |
| `AutoModelForSeq2SeqLM` | Seq2Seq (T5) |
| `AutoModelForSequenceClassification` | Classification |
| `AutoModelForTokenClassification` | NER, POS |
| `AutoModelForQuestionAnswering` | QA |

### Tokenizer Usage
```python
# Encode
inputs = tokenizer("Hello, how are you?", return_tensors="pt")
# → {'input_ids': tensor([[...]]]), 'attention_mask': tensor([[...]]]}

# Batch with padding
inputs = tokenizer(
    ["Hello!", "How are you doing today?"],
    padding=True,
    truncation=True,
    max_length=512,
    return_tensors="pt"
)

# Decode
output_ids = model.generate(**inputs, max_new_tokens=100)
text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
```

### Text Generation Pipeline
```python
from transformers import pipeline

generator = pipeline(
    "text-generation",
    model="meta-llama/Llama-3.1-8B-Instruct",
    torch_dtype=torch.bfloat16,
    device_map="auto"
)

output = generator(
    "Tell me about RAG systems:",
    max_new_tokens=200,
    temperature=0.7,
    top_p=0.9,
    do_sample=True
)
print(output[0]['generated_text'])
```

### Pipeline Tasks
```python
pipe = pipeline("text-classification")         # sentiment
pipe = pipeline("ner")                         # entity recognition
pipe = pipeline("summarization")               # summarization
pipe = pipeline("translation_en_to_fr")        # translation
pipe = pipeline("question-answering")          # QA
pipe = pipeline("feature-extraction")          # embeddings
```

### Trainer API (Fine-tuning)
```python
from transformers import Trainer, TrainingArguments, DataCollatorForSeq2Seq

training_args = TrainingArguments(
    output_dir="./results",
    overwrite_output_dir=True,
    num_train_epochs=3,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    gradient_accumulation_steps=4,
    eval_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-5,
    warmup_ratio=0.03,
    lr_scheduler_type="cosine",
    bf16=True,
    logging_steps=50,
    report_to="wandb",         # experiment tracking
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    tokenizer=tokenizer,
    data_collator=DataCollatorForSeq2Seq(tokenizer),
)
trainer.train()
```

### Hugging Face Datasets
```python
from datasets import load_dataset

# Load public dataset
dataset = load_dataset("squad")
dataset = load_dataset("imdb")
dataset = load_dataset("json", data_files="my_data.jsonl")

# Preprocessing
def tokenize_function(examples):
    return tokenizer(examples["text"], truncation=True, max_length=512)

tokenized_dataset = dataset.map(tokenize_function, batched=True)
```

### Accelerate (Multi-GPU)
```python
from accelerate import Accelerator

accelerator = Accelerator()
model, optimizer, train_dataloader = accelerator.prepare(
    model, optimizer, train_dataloader
)

for batch in train_dataloader:
    with accelerator.accumulate(model):
        outputs = model(**batch)
        loss = outputs.loss
        accelerator.backward(loss)
        optimizer.step()
        optimizer.zero_grad()
```

---

## 4. Other Important Libraries

### LangChain
Framework for building LLM applications (RAG, agents, chains).
```python
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.llms import Ollama

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant."),
    ("user", "{input}")
])
llm = Ollama(model="llama3")
chain = prompt | llm
response = chain.invoke({"input": "What is RAG?"})
```

### vLLM (Inference)
High-throughput LLM serving with PagedAttention.
```python
from vllm import LLM, SamplingParams

llm = LLM(model="meta-llama/Llama-3.1-8B-Instruct")
sampling_params = SamplingParams(temperature=0.7, max_tokens=512)
outputs = llm.generate(["Tell me about transformers"], sampling_params)
```

### Weights & Biases (wandb) — Experiment Tracking
```python
import wandb
wandb.init(project="llm-finetuning", name="lora-run-1")
# Log metrics automatically via Trainer's report_to="wandb"
```

---

## 5. Interview Questions — Frameworks

**Q: What is the difference between PyTorch and TensorFlow?**
> Both support dynamic computation graphs (TF2+). PyTorch is more Pythonic and dominates research. TensorFlow has stronger production deployment tooling (TF Serving, TFLite for mobile). The Hugging Face ecosystem is PyTorch-native, making PyTorch the default choice for LLM work.

**Q: What does `device_map="auto"` do in Hugging Face?**
> It automatically distributes model layers across available GPUs (and CPU/disk if needed) using Accelerate's big model inference. Essential for loading large models that don't fit in a single GPU.

**Q: What is gradient accumulation and why is it used?**
> Accumulate gradients over multiple batches before doing an optimizer step. Simulates a larger effective batch size without requiring more GPU memory. `gradient_accumulation_steps=4` with `batch_size=4` = effective batch of 16.

**Q: What is the `attention_mask` in Hugging Face tokenizers?**
> A binary mask indicating real tokens (1) vs padding tokens (0). The model uses it to ignore padding when computing attention. Required when batching sequences of different lengths.

**Q: What is the difference between `model.train()` and `model.eval()`?**
> `model.train()` enables dropout and batch norm in training mode. `model.eval()` disables them for inference — dropout is bypassed and batch norm uses running statistics instead of batch statistics.

---

## Quick Reference Cheat Sheet

```
PyTorch:    Dominant research framework, dynamic graph, Pythonic
TensorFlow: Google, better production tooling (Serving, Lite)
HuggingFace: Pre-trained models, tokenizers, Trainer, Datasets
PEFT:       LoRA, QLoRA adapter management
TRL:        SFT, DPO, PPO training with Hugging Face
Accelerate: Multi-GPU training made easy
vLLM:       Fast LLM inference with PagedAttention
LangChain:  RAG, agents, chains framework
```
