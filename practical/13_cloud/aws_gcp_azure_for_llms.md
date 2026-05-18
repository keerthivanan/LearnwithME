# 12 — Cloud Platforms: AWS, GCP, Azure for ML

> JD: "Familiarity with cloud-based solutions and tools (AWS, GCP, or Azure) for scalable model training and deployment."

---

## 1. Overview — Three Cloud Providers

**What it is:** The three major cloud platforms each offer their own complete ML ecosystem. Each has a training service, a managed LLM API (so you don't manage GPUs yourself), object storage, and GPU compute instances. You need to know the names and rough equivalences for interviews.

**Analogy:** The three cloud providers are like three different cities. Each city has a train station, an airport, warehouses, and hotels — but they have different names and slightly different layouts. If you know how to get around one city you can quickly adapt to the others.

| Provider | ML Service | LLM Inference | Storage | Compute |
|----------|-----------|--------------|---------|---------|
| **AWS** | SageMaker | Bedrock | S3 | EC2 (p4d, p3) |
| **GCP** | Vertex AI | Vertex AI | GCS | TPUs, A100s |
| **Azure** | Azure ML | Azure OpenAI | Blob Storage | A100s, H100s |

---

## 2. AWS for ML

**What it is:** Amazon Web Services is the largest cloud provider. Its ML ecosystem is built around SageMaker (for training and hosting your own models) and Bedrock (for accessing pre-built LLM APIs without managing infrastructure).

### Core Services

**Amazon SageMaker** — End-to-end ML platform

**What it is:** AWS's managed ML platform. SageMaker handles the underlying infrastructure so you can focus on your training code — it spins up the right GPU instances, mounts your data from S3, runs your training script, saves the model, and tears down the instance when done. No manual instance management.

```python
import sagemaker                             # SageMaker Python SDK
from sagemaker.huggingface import HuggingFace  # HuggingFace estimator for SageMaker

# Define the training job configuration
estimator = HuggingFace(
    entry_point="train.py",                  # your training script file name
    source_dir="./scripts",                  # directory containing your training code
    role=sagemaker.get_execution_role(),     # IAM role that grants SageMaker permission to access S3, ECR etc.
    instance_type="ml.p4d.24xlarge",         # GPU instance type — p4d = 8× A100 80GB GPUs
    instance_count=2,                        # how many of these instances to use (2 × 8 GPUs = 16 GPUs)
    transformers_version="4.36",             # which HuggingFace Transformers version to use
    pytorch_version="2.1",                   # which PyTorch version to use
    py_version="py310",                      # Python version
    hyperparameters={
        "model_id": "meta-llama/Llama-3.1-8B",  # passed to your training script as --model_id
        "epochs": 3,                              # number of training epochs
        "batch_size": 8,                          # per-GPU batch size
        "lora_r": 16,                             # LoRA rank for parameter-efficient fine-tuning
    }
)
estimator.fit({"training": "s3://my-bucket/data/"})  # start the training job, pointing at data in S3
```

**WHY SageMaker over raw EC2:** SageMaker handles provisioning, auto-shutdown after training completes, integration with S3 and model registry, and distributed training setup. With raw EC2 you have to manage all of this yourself.

**Amazon Bedrock** — Managed LLM API service

**What it is:** Instead of running GPU instances yourself, Bedrock gives you API access to many top LLMs (Claude, Llama, Titan) with a simple per-token pricing model. You call an API, you get a response, you pay per token. No servers to manage.

- Access to Claude, Llama, Titan, Mistral models without managing any infrastructure
- Simple API, pay-per-token (scales to zero when not in use)

```python
import boto3   # AWS SDK for Python
import json    # for JSON serialization

bedrock = boto3.client("bedrock-runtime", region_name="us-east-1")  # create Bedrock client for us-east-1 region

# Call Claude via Bedrock — same model, AWS's infrastructure
response = bedrock.invoke_model(
    modelId="anthropic.claude-3-5-sonnet-20241022-v2:0",  # Bedrock's model ID for Claude 3.5 Sonnet
    body=json.dumps({
        "anthropic_version": "bedrock-2023-05-31",        # required API version header
        "max_tokens": 1024,                                # maximum tokens to generate
        "messages": [{"role": "user", "content": "What is RAG?"}]  # conversation messages
    })
)
result = json.loads(response["body"].read())  # deserialize the response body from JSON
print(result["content"][0]["text"])           # extract and print the generated text
```

**WHY Bedrock over calling Anthropic/OpenAI directly:** Everything stays inside the AWS VPC (data never leaves AWS), single billing account, IAM-based access control instead of API keys, and enterprise compliance features (SOC2, HIPAA).

**Amazon S3** — Object storage for model weights and datasets

**What it is:** AWS's massively scalable file storage service. Store anything — training datasets, model checkpoints, final model weights. Extremely cheap (~$0.023/GB/month) and infinitely scalable.

```python
import boto3                                   # AWS SDK

s3 = boto3.client("s3")                        # create S3 client
# Upload model weights to S3
s3.upload_file("model.pt", "my-bucket", "models/model.pt")    # upload local file to S3
# Download model weights from S3
s3.download_file("my-bucket", "models/model.pt", "model.pt")  # download from S3 to local
```

**WHY S3 for ML:** S3 integrates natively with SageMaker — training jobs can read data from S3 and write checkpoints back to S3 directly without copying to local disk. Also survives instance failures (data persists even if your EC2 instance dies).

**Amazon ECR** — Container registry for Docker images

**What it is:** Amazon's Docker container registry. Store your Docker images here, and SageMaker/EKS can pull them directly without any authentication complexity.

**AWS Key Services for LLM Work:**
| Service | Use |
|---------|-----|
| SageMaker | Training, fine-tuning, hosting custom models |
| Bedrock | Managed LLM APIs — no GPU management |
| S3 | Data and model weight storage |
| ECR | Docker image registry |
| EKS | Kubernetes for container orchestration |
| EC2 (p4d/p3) | Raw GPU instances (when you need maximum control) |
| Lambda | Serverless inference for very small models or pre/post processing |

### SageMaker Instance Types for LLM

**What it is:** A reference guide to the GPU instance types available on SageMaker. Choose based on model size and whether you are training or just doing inference.

| Instance | GPUs | VRAM | Use Case |
|----------|------|------|---------|
| ml.g5.2xlarge | 1× A10G | 24GB | Small model inference (7B in INT4) |
| ml.g5.48xlarge | 8× A10G | 192GB | 70B inference |
| ml.p4d.24xlarge | 8× A100 | 320GB | Large model training |
| ml.p4de.24xlarge | 8× A100 80GB | 640GB | Very large training (70B+ in BF16) |

---

## 3. GCP for ML

**What it is:** Google Cloud Platform is the birthplace of Transformers, TensorFlow, and TPUs. Its ML ecosystem is built around Vertex AI and is particularly strong for TPU-based training of very large models.

### Core Services

**Vertex AI** — Google's end-to-end ML platform

**What it is:** GCP's equivalent of AWS SageMaker — a managed platform for training and deploying ML models. It handles infrastructure, integrates with GCS for data, and provides a model registry.

```python
from google.cloud import aiplatform  # Google Cloud AI Platform SDK

aiplatform.init(project="my-project", location="us-central1")  # initialize the SDK with your project and region

# Define and submit a custom training job
job = aiplatform.CustomTrainingJob(
    display_name="llm-finetuning",          # human-readable name for this job
    script_path="train.py",                 # your training script
    requirements=["transformers==4.36", "peft", "trl"],  # pip packages to install in the training container
    container_uri="gcr.io/deeplearning-platform-release/pytorch-gpu.2-1",  # Docker base image from Google
)

job.run(
    machine_type="a2-highgpu-8g",           # machine type — a2-highgpu = A100 GPUs, 8g = 8 GPUs
    accelerator_type="NVIDIA_TESLA_A100",   # GPU type
    accelerator_count=8,                    # number of GPUs
    args=["--model_id", "meta-llama/Llama-3.1-8B"],  # arguments passed to your train.py
)
```

**Google Cloud Storage (GCS)** — Object storage

**What it is:** GCP's equivalent of S3. Same concept — store anything, scales infinitely, integrates with Vertex AI for training data and checkpoints.

```python
from google.cloud import storage   # Google Cloud Storage SDK

client = storage.Client()                       # create storage client
bucket = client.bucket("my-bucket")            # reference a specific bucket
blob = bucket.blob("models/model.pt")           # reference a specific file (blob)
blob.upload_from_filename("model.pt")           # upload local file to GCS
```

**TPUs** — Google's custom ML accelerators

**What it is:** Tensor Processing Units — custom silicon chips designed by Google specifically for neural network training. They have very high memory bandwidth and are especially fast for matrix multiplication (the core operation in neural networks). Used to train T5, PaLM, and Gemini.

**Analogy:** TPUs are like specialized racing cars built only for one track. On their specific track (large-scale matrix multiplication for neural networks), they are faster than general-purpose GPUs. Outside that use case, they are not useful.

- Faster than A100s for certain large-scale training workloads
- Used to train T5, PaLM, Gemini
- Requires PyTorch/XLA (a special bridge library) for PyTorch models

```python
import torch_xla.core.xla_model as xm    # XLA (Accelerated Linear Algebra) model utilities
device = xm.xla_device()                  # get the TPU device (like torch.device("cuda") but for TPU)
model = model.to(device)                  # move model to TPU memory
```

**GCP Key Services:**
| Service | Use |
|---------|-----|
| Vertex AI | Training, fine-tuning, hosting |
| Cloud Storage (GCS) | Data and model storage |
| Artifact Registry | Docker image registry (equivalent to ECR) |
| GKE | Kubernetes (equivalent to EKS) |
| Cloud TPU | TPU-based training at massive scale |
| A2/A3 VMs | A100/H100 GPU instances |

---

## 4. Azure for ML

**What it is:** Microsoft Azure's ML ecosystem is particularly strong for enterprises already using Microsoft products. Its key differentiator is Azure OpenAI Service — the only way to access GPT-4 through Microsoft's enterprise-grade infrastructure with compliance guarantees.

### Core Services

**Azure Machine Learning** — End-to-end ML platform

**What it is:** Azure's equivalent of SageMaker and Vertex AI. Manage training jobs, compute clusters, and model deployments through either a UI or SDK.

```python
from azure.ai.ml import MLClient, command               # Azure ML SDK
from azure.ai.ml.entities import AmlCompute             # compute cluster entity
from azure.identity import DefaultAzureCredential       # handles Azure authentication automatically

# Authenticate and connect to your Azure ML workspace
ml_client = MLClient(
    credential=DefaultAzureCredential(),                # auto-detects authentication method
    subscription_id="<subscription-id>",               # your Azure subscription
    resource_group_name="<resource-group>",            # your resource group
    workspace_name="<workspace-name>"                  # your Azure ML workspace
)

# Define the training job
job = command(
    code="./scripts",                                   # local directory with your code
    command="python train.py --model_id meta-llama/Llama-3.1-8B",  # command to run in the container
    environment="azureml:AzureML-PyTorch-2.1-GPU:1",   # pre-built Azure ML environment with PyTorch + GPU
    compute="gpu-cluster",                              # name of your configured GPU compute cluster
    instance_type="Standard_ND96asr_v4",               # specific VM size: 8× A100 GPUs
)
returned_job = ml_client.jobs.create_or_update(job)    # submit the job to Azure ML
```

**Azure OpenAI Service** — Enterprise-grade access to GPT-4 and other OpenAI models

**What it is:** Microsoft's enterprise version of the OpenAI API. Same models (GPT-4, GPT-4o) but running within Azure's infrastructure — key for enterprises that need data residency, compliance, and private endpoints.

```python
from openai import AzureOpenAI  # OpenAI SDK's Azure variant

# Connect to your Azure OpenAI endpoint
client = AzureOpenAI(
    api_key="your-key",                                         # your Azure OpenAI API key
    azure_endpoint="https://your-endpoint.openai.azure.com/",  # your specific Azure endpoint URL
    api_version="2024-05-01-preview"                           # API version to use
)

# Call GPT-4 — same code as regular OpenAI but data stays in your Azure tenant
response = client.chat.completions.create(
    model="gpt-4o",                                            # your deployed model name in Azure
    messages=[{"role": "user", "content": "Explain transformers"}]  # messages to send
)
```

**WHY Azure OpenAI over regular OpenAI:** Data never leaves your Azure tenant (critical for HIPAA/GDPR compliance). Private VNet endpoints, no public internet. Enterprise SLAs, support contracts, and Microsoft's compliance certifications.

**Azure Blob Storage** — Object storage

**What it is:** Azure's equivalent of S3 and GCS — object storage for training data, model weights, and checkpoints.

```python
from azure.storage.blob import BlobServiceClient  # Azure Storage SDK

# Connect using a connection string
client = BlobServiceClient.from_connection_string("<connection-string>")  # authenticate to storage account
container_client = client.get_container_client("models")  # get a container (like an S3 bucket)
with open("model.pt", "rb") as f:                         # open local file in binary read mode
    container_client.upload_blob("model.pt", f)           # upload to Azure Blob Storage
```

---

## 5. Common Cloud Patterns for LLM Work

**What it is:** The standard workflows for training and deploying LLMs on any cloud provider. These patterns are the same regardless of which cloud you use — only the service names differ.

### Training Pipeline

**What it is:** The end-to-end flow from raw data to a trained model ready for deployment.

```
1. Store dataset in S3/GCS/Blob           ← put training data in durable cloud storage
2. Launch training job (SageMaker/Vertex/Azure ML)  ← managed service spins up GPUs
3. Save checkpoints periodically to object storage   ← fault tolerance (resume on failure)
4. Training completes → final model in object storage  ← model artifacts saved durably
5. Register model in model registry (MLflow/SageMaker/Vertex)  ← version and tag the model
```

### Inference Pipeline

**What it is:** The end-to-end flow from a trained model to a live API endpoint serving users.

```
1. Load model from registry               ← retrieve specific version of the model
2. Package in Docker container            ← bundle model + serving code + dependencies
3. Push to ECR/Artifact Registry          ← store image in cloud container registry
4. Deploy to EKS/GKE/AKS (Kubernetes)    ← run containers with auto-scaling and load balancing
5. Auto-scale based on GPU metrics        ← add/remove instances based on load
6. API gateway in front for auth + rate limiting  ← security and traffic management layer
```

### Cost Management

**What it is:** Practical techniques to reduce your cloud bill for LLM training and inference. GPU compute is expensive — these optimizations can cut costs dramatically.

- Use **spot/preemptible instances** for training (70-90% cheaper than on-demand — they can be interrupted but you have checkpoints)
- Use **auto-scaling** to zero out when not in use (if inference is only needed during business hours, scale to 0 overnight)
- Use **quantization** to reduce GPU memory and fit more on cheaper GPUs (7B in INT4 runs on a $0.80/hr GPU instead of a $3.50/hr one)

**WHY spot instances are safe for training:** Training is resumable. If your spot instance is terminated, you restart from the last checkpoint. For inference (user requests), spot instances are not appropriate since interruptions cause request failures.

---

## 6. Managed LLM APIs (No Infrastructure)

**What it is:** When you want to use an LLM without managing any servers, GPUs, or infrastructure. You call an API, get a response, pay per token. Best for low-to-medium traffic or when you cannot justify self-hosting.

When you don't want to manage GPU infrastructure:

| Provider | Service | Models Available |
|----------|---------|-----------------|
| AWS | Bedrock | Claude, Llama, Titan, Mistral |
| GCP | Vertex AI | Gemini, Claude, Llama |
| Azure | Azure OpenAI | GPT-4, GPT-4o, Embeddings |
| OpenAI | API | GPT-4, GPT-4o, DALL-E |
| Anthropic | API | Claude 3/4 family |
| Together AI | API | Open-source LLMs (Llama, Mistral, etc.) |

---

## AWS SageMaker — Deploying a Model as an HTTP Endpoint

**What it is:** SageMaker can deploy your model as a persistent HTTP endpoint that auto-scales based on traffic and can even scale to zero when idle (saving cost when not in use).

```python
import boto3                                              # AWS SDK
from sagemaker.huggingface import HuggingFaceModel       # SageMaker HuggingFace model class

# Define the model to deploy
huggingface_model = HuggingFaceModel(
    model_data="s3://my-bucket/model.tar.gz",            # your model weights stored in S3
    role="arn:aws:iam::123:role/SageMakerRole",          # IAM role with S3 and ECR access
    transformers_version="4.37",                          # HuggingFace version in the container
    pytorch_version="2.1",                                # PyTorch version
    py_version="py310",                                   # Python version
    env={
        "HF_MODEL_ID": "meta-llama/Llama-3.1-8B",        # model to load from HF Hub (or use model_data)
        "SM_NUM_GPUS": "1"                                # how many GPUs the container should use
    }
)

# Deploy the model — SageMaker creates an HTTP endpoint
predictor = huggingface_model.deploy(
    initial_instance_count=1,                             # start with 1 instance
    instance_type="ml.g5.2xlarge",                        # instance type (1x A10G GPU, 24GB VRAM)
    endpoint_name="llama-3-endpoint"                      # name of the HTTP endpoint
)

# Call the endpoint — SageMaker routes the HTTP request to your model
response = predictor.predict({"inputs": "What is machine learning?"})

# Configure auto-scaling — scale to 0 when idle, scale up to 5 under load
import boto3
client = boto3.client("application-autoscaling")         # auto-scaling API client
client.register_scalable_target(
    ServiceNamespace="sagemaker",                        # tell auto-scaler we're scaling SageMaker
    ResourceId=f"endpoint/llama-3-endpoint/variant/AllTraffic",  # which endpoint variant to scale
    ScalableDimension="sagemaker:variant:DesiredInstanceCount",  # we're scaling instance count
    MinCapacity=0,                                        # scale to zero when idle (no traffic = no cost)
    MaxCapacity=5                                         # maximum 5 instances under peak load
)
```

**WHY scale to zero:** An endpoint that runs 24/7 even with no traffic costs money constantly. Scale-to-zero means you only pay when requests are actually being served. Cold start latency (time to spin up from 0) is the trade-off — acceptable for non-latency-critical workloads.

---

## IAM Security — Non-Negotiable in Production

**What it is:** Identity and Access Management (IAM) controls who (or which service) can do what in your cloud environment. Getting this wrong is a major security risk — over-permissioned services are the #1 cause of cloud security incidents.

**Analogy:** IAM is like key cards in an office building. Each employee (service) gets a key card that only opens the rooms they need for their job. A janitor's key card opens cleaning supply rooms — not the CEO's office. The principle: only give access to exactly what is needed.

Least Privilege Principle: every service gets ONLY the permissions it needs, nothing more.

Bad (never do this):
```json
{
  "Effect": "Allow",
  "Action": "*",          // allows ALL actions in ALL AWS services
  "Resource": "*"         // on ALL resources — this is admin access to your entire AWS account
}
```

Good — SageMaker training job policy:
```json
{
  "Effect": "Allow",
  "Action": [
    "s3:GetObject",                   // can only read from S3
    "s3:PutObject",                   // can only write to S3 (for checkpoints)
    "ecr:GetAuthorizationToken",      // can authenticate to pull Docker images
    "cloudwatch:PutMetricData"        // can write metrics to CloudWatch for monitoring
  ],
  "Resource": [
    "arn:aws:s3:::my-training-bucket/*",        // only this specific S3 bucket
    "arn:aws:ecr:::repository/my-training-image" // only this specific ECR repository
  ]
}
```

Secrets management (NEVER hardcode credentials — this is how companies get hacked):
```python
# BAD — credentials in source code, visible in git history:
api_key = "sk-abc123..."  # NEVER DO THIS — this ends up in your git repo

# GOOD — AWS Secrets Manager stores secrets securely:
import boto3, json
client = boto3.client("secretsmanager")                        # Secrets Manager client
secret = client.get_secret_value(SecretId="prod/openai-api-key")  # retrieve by name
api_key = json.loads(secret["SecretString"])["api_key"]        # extract the key value

# OR use environment variables injected at runtime (second best):
import os
api_key = os.environ["OPENAI_API_KEY"]  # read from environment, never store in code
```

VPC private endpoints:
- LLM inference should run in a private subnet (no public internet access)
- SageMaker endpoint in VPC: data never traverses the public internet
- Critical for HIPAA, SOC2, GDPR compliance where data residency and encryption in transit are mandatory

**Interview: "How do you secure an ML system on AWS?"** → "IAM least privilege — training jobs only access their specific S3 bucket. Secrets in AWS Secrets Manager, never in code. VPC private endpoints so model traffic never traverses public internet. CloudTrail for audit logging of all API calls."

---

## MLOps Pipelines — Automated Retraining and Deployment

**What it is:** A complete automated system that handles the full cycle from new data to updated model in production, with quality gates to prevent bad models from reaching users.

**Analogy:** An MLOps pipeline is like a car factory's quality control system. Raw materials (new data) come in, cars (models) are manufactured (trained), each car goes through quality checks (evaluation), only cars that pass ship to dealers (deployment), and the factory monitors performance after shipping (monitoring).

Manual process (bad): train → evaluate → manually deploy → monitor → repeat
MLOps pipeline (good): automated trigger → train → evaluate → quality gate → deploy → monitor → retrigger

AWS SageMaker Pipelines:
```python
from sagemaker.workflow.pipeline import Pipeline             # SageMaker Pipeline class
from sagemaker.workflow.steps import TrainingStep, ProcessingStep  # pipeline step types

# Define pipeline steps — each step is a unit of work
preprocess = ProcessingStep(name="Preprocess", processor=sklearn_processor)  # data preprocessing step
train = TrainingStep(name="Train", estimator=huggingface_estimator)           # model training step
evaluate = ProcessingStep(name="Evaluate", depends_on=[train])                # evaluation runs after training

# Quality gate: only deploy if the model meets the quality threshold
from sagemaker.workflow.conditions import ConditionGreaterThanOrEqualTo       # comparison condition
from sagemaker.workflow.condition_step import ConditionStep                   # conditional pipeline step

condition = ConditionGreaterThanOrEqualTo(
    left=JsonGet(step=evaluate, property_file="metrics.json", json_path="f1_score"),  # read F1 score from evaluation output
    right=0.85    # threshold — only continue if F1 >= 0.85
)
deploy_step = ConditionStep(
    name="CheckQuality",
    conditions=[condition],  # apply the F1 threshold condition
    if_steps=[deploy],       # if condition passes: deploy to production
    else_steps=[fail]        # if condition fails: mark pipeline run as failed, do NOT deploy
)

# Assemble all steps into the pipeline
pipeline = Pipeline(
    name="LLM-Training-Pipeline",
    steps=[preprocess, train, evaluate, deploy_step]  # steps run in dependency order
)
pipeline.upsert(role_arn=role)  # create or update the pipeline definition in SageMaker

# Trigger: run pipeline when new training data arrives in S3
# → EventBridge rule: S3 PutObject → Lambda → pipeline.start()
```

**WHY quality gates are critical:** Without a quality gate, a bad training run (wrong hyperparameters, corrupt data, code bug) can automatically deploy a broken model to production. The quality gate stops this — a human never has to review routine updates, but bad models are automatically blocked.

CI/CD for models:
- Code change → GitHub Actions → build Docker image → push to ECR → run pipeline
- Data change → S3 event notification → trigger retraining pipeline
- Model degradation detected in production → alert → trigger retraining pipeline

**Interview: "How do you automate model retraining?"** → "MLOps pipeline with automated triggers (new data, scheduled, or drift detection), automated evaluation with quality gates (don't deploy if metrics drop), and automated deployment on pass. SageMaker Pipelines, Vertex AI Pipelines, or Azure ML Pipelines are the managed options."

---

## 7. Interview Questions — Cloud

**Q: What AWS service would you use to train a large LLM?**
> Amazon SageMaker for managed training (handles infrastructure, distributed training), or raw EC2 p4d instances for more control. Use S3 for data and checkpoints, and integrate with SageMaker Model Registry for versioning.

**Q: What is the difference between AWS SageMaker and Bedrock?**
> SageMaker is for training, fine-tuning, and deploying your own models with full infrastructure control. Bedrock is a managed service that gives API access to pre-built models (Claude, Llama, Titan) without managing any infrastructure — pay per token.

**Q: How would you reduce cloud costs for LLM training?**
> Use spot/preemptible instances (70-90% cheaper with checkpointing for fault tolerance). Use quantization (QLoRA) to train on smaller/fewer GPUs. Use gradient accumulation to maximize GPU utilization. Use managed services like Bedrock for inference instead of running your own GPU servers when traffic is low.

**Q: What are Google TPUs and when would you use them?**
> TPUs (Tensor Processing Units) are Google's custom ASIC accelerators optimized for matrix operations in neural networks. They're faster than A100s for certain large-scale training workloads (T5, PaLM, Gemini were trained on TPUs). Use PyTorch/XLA for PyTorch on TPU. Best for: training very large models from scratch at Google-scale where the matrix operations dominate computation.

---

## Quick Reference Cheat Sheet

```
AWS:            SageMaker (train/deploy) + Bedrock (managed API) + S3 (storage)
GCP:            Vertex AI (train/deploy) + TPUs (scale training) + GCS (storage)
Azure:          Azure ML (train/deploy) + Azure OpenAI (managed GPT-4) + Blob
Spot instances: 70-90% cheaper — use with checkpointing for fault tolerance
Managed APIs:   Bedrock/Vertex/Azure OpenAI — no GPU management, pay per token
IAM:            Least privilege — each service only accesses what it needs
Secrets:        AWS Secrets Manager / env vars — NEVER hardcode credentials
MLOps pipeline: Automated train → evaluate → quality gate → deploy → monitor
Scale to zero:  Auto-scale SageMaker endpoints to 0 when idle — no traffic = no cost
VPC:            Private subnets for compliance (HIPAA, GDPR, SOC2)
```
