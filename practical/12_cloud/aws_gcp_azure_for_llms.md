# 12 — Cloud Platforms: AWS, GCP, Azure for ML

> JD: "Familiarity with cloud-based solutions and tools (AWS, GCP, or Azure) for scalable model training and deployment."

---

## 1. Overview — Three Cloud Providers

| Provider | ML Service | LLM Inference | Storage | Compute |
|----------|-----------|--------------|---------|---------|
| **AWS** | SageMaker | Bedrock | S3 | EC2 (p4d, p3) |
| **GCP** | Vertex AI | Vertex AI | GCS | TPUs, A100s |
| **Azure** | Azure ML | Azure OpenAI | Blob Storage | A100s, H100s |

---

## 2. AWS for ML

### Core Services

**Amazon SageMaker** — End-to-end ML platform
```python
import sagemaker
from sagemaker.huggingface import HuggingFace

# Fine-tuning job on SageMaker
estimator = HuggingFace(
    entry_point="train.py",
    source_dir="./scripts",
    role=sagemaker.get_execution_role(),
    instance_type="ml.p4d.24xlarge",   # 8× A100 GPUs
    instance_count=2,
    transformers_version="4.36",
    pytorch_version="2.1",
    py_version="py310",
    hyperparameters={
        "model_id": "meta-llama/Llama-3.1-8B",
        "epochs": 3,
        "batch_size": 8,
        "lora_r": 16,
    }
)
estimator.fit({"training": "s3://my-bucket/data/"})
```

**Amazon Bedrock** — Managed LLM API service
- Access to Claude, Llama, Titan, Mistral models without managing infrastructure
- Simple API, pay-per-token

```python
import boto3
import json

bedrock = boto3.client("bedrock-runtime", region_name="us-east-1")

response = bedrock.invoke_model(
    modelId="anthropic.claude-3-5-sonnet-20241022-v2:0",
    body=json.dumps({
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": 1024,
        "messages": [{"role": "user", "content": "What is RAG?"}]
    })
)
result = json.loads(response["body"].read())
print(result["content"][0]["text"])
```

**Amazon S3** — Object storage for model weights and datasets
```python
import boto3

s3 = boto3.client("s3")
# Upload model
s3.upload_file("model.pt", "my-bucket", "models/model.pt")
# Download
s3.download_file("my-bucket", "models/model.pt", "model.pt")
```

**Amazon ECR** — Container registry for Docker images

**AWS Key Services for LLM Work:**
| Service | Use |
|---------|-----|
| SageMaker | Training, fine-tuning, hosting |
| Bedrock | Managed LLM APIs |
| S3 | Data and model storage |
| ECR | Docker image registry |
| EKS | Kubernetes for deployment |
| EC2 (p4d/p3) | Raw GPU instances |
| Lambda | Serverless inference for small models |

### SageMaker Instance Types for LLM
| Instance | GPUs | VRAM | Use Case |
|----------|------|------|---------|
| ml.g5.2xlarge | 1× A10G | 24GB | Small model inference |
| ml.g5.48xlarge | 8× A10G | 192GB | 70B inference |
| ml.p4d.24xlarge | 8× A100 | 320GB | Large model training |
| ml.p4de.24xlarge | 8× A100 80GB | 640GB | Very large training |

---

## 3. GCP for ML

### Core Services

**Vertex AI** — Google's end-to-end ML platform
```python
from google.cloud import aiplatform

aiplatform.init(project="my-project", location="us-central1")

# Create training job
job = aiplatform.CustomTrainingJob(
    display_name="llm-finetuning",
    script_path="train.py",
    requirements=["transformers==4.36", "peft", "trl"],
    container_uri="gcr.io/deeplearning-platform-release/pytorch-gpu.2-1",
)

job.run(
    machine_type="a2-highgpu-8g",  # 8× A100 GPUs
    accelerator_type="NVIDIA_TESLA_A100",
    accelerator_count=8,
    args=["--model_id", "meta-llama/Llama-3.1-8B"],
)
```

**Google Cloud Storage (GCS)** — Object storage
```python
from google.cloud import storage

client = storage.Client()
bucket = client.bucket("my-bucket")
blob = bucket.blob("models/model.pt")
blob.upload_from_filename("model.pt")
```

**TPUs** — Google's custom ML accelerators
- Faster than A100s for certain operations (especially matrix multiplication)
- Used to train T5, PaLM, Gemini
- PyTorch/XLA for PyTorch on TPU

```python
import torch_xla.core.xla_model as xm
device = xm.xla_device()
model = model.to(device)
```

**GCP Key Services:**
| Service | Use |
|---------|-----|
| Vertex AI | Training, fine-tuning, hosting |
| Cloud Storage | Data and model storage |
| Artifact Registry | Docker images |
| GKE | Kubernetes |
| Cloud TPU | TPU-based training |
| A2/A3 VMs | A100/H100 GPU instances |

---

## 4. Azure for ML

### Core Services

**Azure Machine Learning** — End-to-end ML platform
```python
from azure.ai.ml import MLClient, command
from azure.ai.ml.entities import AmlCompute
from azure.identity import DefaultAzureCredential

ml_client = MLClient(
    credential=DefaultAzureCredential(),
    subscription_id="<subscription-id>",
    resource_group_name="<resource-group>",
    workspace_name="<workspace-name>"
)

# Submit training job
job = command(
    code="./scripts",
    command="python train.py --model_id meta-llama/Llama-3.1-8B",
    environment="azureml:AzureML-PyTorch-2.1-GPU:1",
    compute="gpu-cluster",
    instance_type="Standard_ND96asr_v4",  # 8× A100 GPUs
)
returned_job = ml_client.jobs.create_or_update(job)
```

**Azure OpenAI Service** — Enterprise-grade access to GPT-4 and other OpenAI models
```python
from openai import AzureOpenAI

client = AzureOpenAI(
    api_key="your-key",
    azure_endpoint="https://your-endpoint.openai.azure.com/",
    api_version="2024-05-01-preview"
)

response = client.chat.completions.create(
    model="gpt-4o",
    messages=[{"role": "user", "content": "Explain transformers"}]
)
```

**Azure Blob Storage** — Object storage
```python
from azure.storage.blob import BlobServiceClient

client = BlobServiceClient.from_connection_string("<connection-string>")
container_client = client.get_container_client("models")
with open("model.pt", "rb") as f:
    container_client.upload_blob("model.pt", f)
```

---

## 5. Common Cloud Patterns for LLM Work

### Training Pipeline
```
1. Store dataset in S3/GCS/Blob
2. Launch training job (SageMaker/Vertex/Azure ML)
3. Save checkpoints periodically to object storage
4. Training completes → final model in object storage
5. Register model in model registry (MLflow/SageMaker/Vertex)
```

### Inference Pipeline
```
1. Load model from registry
2. Package in Docker container
3. Push to ECR/Artifact Registry
4. Deploy to EKS/GKE/AKS (Kubernetes)
5. Auto-scale based on GPU metrics
6. API gateway in front for auth + rate limiting
```

### Cost Management
- Use **spot/preemptible instances** for training (70-90% cheaper)
- Use **auto-scaling** to zero out when not in use
- Use **quantization** to reduce GPU memory and fit more on cheaper GPUs

---

## 6. Managed LLM APIs (No Infrastructure)

When you don't want to manage GPU infrastructure:

| Provider | Service | Models Available |
|----------|---------|-----------------|
| AWS | Bedrock | Claude, Llama, Titan, Mistral |
| GCP | Vertex AI | Gemini, Claude, Llama |
| Azure | Azure OpenAI | GPT-4, GPT-4o, Embeddings |
| OpenAI | API | GPT-4, GPT-4o, DALL-E |
| Anthropic | API | Claude 3/4 family |
| Together AI | API | Open-source LLMs |

---

## 7. Interview Questions — Cloud

**Q: What AWS service would you use to train a large LLM?**
> Amazon SageMaker for managed training (handles infrastructure, distributed training), or raw EC2 p4d instances for more control. Use S3 for data and checkpoints, and integrate with SageMaker Model Registry for versioning.

**Q: What is the difference between AWS SageMaker and Bedrock?**
> SageMaker is for training, fine-tuning, and deploying your own models with full infrastructure control. Bedrock is a managed service that gives API access to pre-built models (Claude, Llama, Titan) without managing any infrastructure — pay per token.

**Q: How would you reduce cloud costs for LLM training?**
> Use spot/preemptible instances (70-90% cheaper with checkpointing for fault tolerance). Use quantization (QLoRA) to train on smaller/fewer GPUs. Use gradient accumulation to maximize GPU utilization. Use managed services like Bedrock for inference instead of running your own GPU servers when traffic is low.

**Q: What are Google TPUs and when would you use them?**
> TPUs (Tensor Processing Units) are Google's custom ASIC accelerators optimized for matrix operations in neural networks. They're faster than A100s for certain large-scale training workloads (T5, PaLM, Gemini were trained on TPUs). Use PyTorch/XLA for PyTorch on TPU.

---

## Quick Reference Cheat Sheet

```
AWS:      SageMaker (train/deploy) + Bedrock (managed API) + S3 (storage)
GCP:      Vertex AI (train/deploy) + TPUs (scale training) + GCS (storage)
Azure:    Azure ML (train/deploy) + Azure OpenAI (managed GPT-4) + Blob
Spot instances:  70-90% cheaper, use with checkpointing
Managed APIs:    Bedrock/Vertex/Azure OpenAI — no GPU management needed
```
