"""
AWS + GCP — Definitions + Code for ML Engineers
=================================================
pip install boto3 google-cloud-storage google-cloud-aiplatform sagemaker joblib

WHAT IS CLOUD COMPUTING?
  → Rent compute, storage, and services over the internet — no physical servers
  → Pay-as-you-go: only pay for what you use

AWS vs GCP:
  → AWS (Amazon Web Services): largest market share, most services
    - S3       : object/file storage (like a hard drive in the cloud)
    - EC2      : virtual machines (Linux/Windows servers on demand)
    - SageMaker: managed ML training + deployment
    - Lambda   : serverless functions (run code without managing servers)
  → GCP (Google Cloud): Google's cloud, known for ML/AI tools
    - GCS      : Google Cloud Storage (like S3)
    - Vertex AI: Google's managed ML platform (like SageMaker)
    - BigQuery : serverless data warehouse (SQL at petabyte scale)
"""

import boto3
import json
import os
import io
import joblib
import pandas as pd


# ══════════════════════════════════════════════════════
# 1. S3 — Object Storage
# ══════════════════════════════════════════════════════
# WHAT IS S3?
#   → Simple Storage Service — store ANY file in the cloud
#   → Organized as: Bucket (folder) → Key (file path)
#   → Global, infinitely scalable, 99.999999999% durability
#
# KEY OPERATIONS:
#   → upload_file() / put_object() : upload to S3
#   → download_file() / get_object(): download from S3
#   → list_objects_v2()            : list files in a bucket
#   → delete_object()              : delete a file
#   → generate_presigned_url()     : create temporary shareable link
#
# USE CASES FOR ML:
#   → Store datasets (CSV, Parquet)
#   → Store trained models (.pkl, .pt, .h5)
#   → Store training logs and artifacts
#   → SageMaker reads training data from S3
#
# AUTHENTICATION:
#   → AWS_ACCESS_KEY_ID + AWS_SECRET_ACCESS_KEY (env vars)
#   → Or: aws configure (CLI)
#   → Or: IAM Role (best for EC2/SageMaker — no keys needed!)

print("=" * 55)
print("1. AWS S3")
print("=" * 55)

s3     = boto3.client("s3", region_name="ap-south-1")
BUCKET = "my-ml-bucket"

def create_bucket(bucket_name, region="ap-south-1"):
    s3.create_bucket(
        Bucket=bucket_name,
        CreateBucketConfiguration={"LocationConstraint": region}
    )
    print(f"  Bucket created: {bucket_name}")

def upload_file(local_path: str, s3_key: str, bucket: str = BUCKET):
    s3.upload_file(local_path, bucket, s3_key)
    print(f"  Uploaded: {local_path} → s3://{bucket}/{s3_key}")

def upload_bytes(data: bytes, s3_key: str, bucket: str = BUCKET):
    s3.put_object(Body=data, Bucket=bucket, Key=s3_key)

def download_file(s3_key: str, local_path: str, bucket: str = BUCKET):
    s3.download_file(bucket, s3_key, local_path)
    print(f"  Downloaded: s3://{bucket}/{s3_key} → {local_path}")

def read_s3_file(s3_key: str, bucket: str = BUCKET) -> bytes:
    obj = s3.get_object(Bucket=bucket, Key=s3_key)
    return obj["Body"].read()

def list_files(prefix: str = "", bucket: str = BUCKET):
    response = s3.list_objects_v2(Bucket=bucket, Prefix=prefix)
    return [obj["Key"] for obj in response.get("Contents", [])]

def delete_file(s3_key: str, bucket: str = BUCKET):
    s3.delete_object(Bucket=bucket, Key=s3_key)

def get_presigned_url(s3_key: str, expires_in: int = 3600, bucket: str = BUCKET):
    url = s3.generate_presigned_url(
        "get_object",
        Params={"Bucket": bucket, "Key": s3_key},
        ExpiresIn=expires_in   # seconds — link expires after this
    )
    return url   # sharable download link, valid for 1 hour

# Save/load ML model directly to/from S3 (no temp file!)
def save_model_to_s3(model, s3_key: str, bucket: str = BUCKET):
    buffer = io.BytesIO()
    joblib.dump(model, buffer)
    buffer.seek(0)
    s3.put_object(Body=buffer.getvalue(), Bucket=bucket, Key=s3_key)
    print(f"  Model saved to s3://{bucket}/{s3_key}")

def load_model_from_s3(s3_key: str, bucket: str = BUCKET):
    obj    = s3.get_object(Bucket=bucket, Key=s3_key)
    buffer = io.BytesIO(obj["Body"].read())
    return joblib.load(buffer)

# Save/load DataFrame to/from S3
def save_df_to_s3(df: pd.DataFrame, s3_key: str, bucket: str = BUCKET):
    csv_buffer = io.StringIO()
    df.to_csv(csv_buffer, index=False)
    s3.put_object(Body=csv_buffer.getvalue(), Bucket=bucket, Key=s3_key)

def load_df_from_s3(s3_key: str, bucket: str = BUCKET) -> pd.DataFrame:
    obj = s3.get_object(Bucket=bucket, Key=s3_key)
    return pd.read_csv(obj["Body"])

print("S3 helper functions defined. Set AWS_ACCESS_KEY_ID + AWS_SECRET_ACCESS_KEY to use.")


# ══════════════════════════════════════════════════════
# 2. EC2 — Virtual Machines
# ══════════════════════════════════════════════════════
# WHAT IS EC2?
#   → Elastic Compute Cloud — rent virtual machines of any size
#   → Choose: CPU type, RAM, GPU (for deep learning), storage
#   → Common ML instance types:
#     - t3.medium  : 2 vCPU, 4GB RAM  — dev/testing
#     - m5.xlarge  : 4 vCPU, 16GB RAM — training on small data
#     - p3.2xlarge : 8 vCPU, 61GB RAM + 1 Tesla V100 GPU — deep learning!
#
# KEY OPERATIONS:
#   → run_instances()       : launch a new VM (like turning on a PC in cloud)
#   → stop_instances()      : stop VM (saves state, still billed for storage)
#   → terminate_instances() : destroy VM permanently (stops billing)
#
# UserData:
#   → Shell script that runs automatically when VM starts
#   → Use to: install packages, download code, start training

print("\n" + "=" * 55)
print("2. AWS EC2")
print("=" * 55)

ec2 = boto3.client("ec2", region_name="ap-south-1")

def launch_instance(instance_type="t3.medium", ami_id="ami-0c55b159cbfafe1f0"):
    response = ec2.run_instances(
        ImageId=ami_id,
        InstanceType=instance_type,
        MinCount=1,
        MaxCount=1,
        KeyName="my-key-pair",
        SecurityGroupIds=["sg-xxxxxxxx"],
        SubnetId="subnet-xxxxxxxx",
        TagSpecifications=[{
            "ResourceType": "instance",
            "Tags": [{"Key": "Name", "Value": "ml-training-server"}]
        }],
        UserData="""#!/bin/bash
            apt-get update
            pip3 install torch scikit-learn pandas
            cd /home/ubuntu && python3 train.py
        """
    )
    instance_id = response["Instances"][0]["InstanceId"]
    print(f"  Launched: {instance_id}")
    return instance_id

def stop_instance(instance_id: str):
    ec2.stop_instances(InstanceIds=[instance_id])

def terminate_instance(instance_id: str):
    ec2.terminate_instances(InstanceIds=[instance_id])

def get_instance_info(instance_id: str):
    response = ec2.describe_instances(InstanceIds=[instance_id])
    inst = response["Reservations"][0]["Instances"][0]
    return {
        "state":     inst["State"]["Name"],
        "public_ip": inst.get("PublicIpAddress"),
        "type":      inst["InstanceType"],
    }

print("EC2 helper functions defined.")


# ══════════════════════════════════════════════════════
# 3. SAGEMAKER — Managed ML Training & Deployment
# ══════════════════════════════════════════════════════
# WHAT IS SAGEMAKER?
#   → AWS's fully managed ML platform — no server management needed
#   → Handles: spinning up VMs, running training, saving models, deploying APIs
#
# TRAINING JOB:
#   → You write a training script (train.py)
#   → SageMaker launches a VM, runs your script, saves model to S3
#   → estimator.fit({"train": "s3://..."}) → triggers training job
#
# ENDPOINT (Deployment):
#   → estimator.deploy() → creates a REST API around your model
#   → Auto-scales: 1 instance when quiet, 5 when traffic spikes
#   → IMPORTANT: delete endpoint when done → stops billing!
#
# BATCH TRANSFORM:
#   → For large datasets (can't fit in real-time)
#   → Runs predictions on S3 files → saves results to S3
#   → Cheaper than endpoints for batch workloads

print("\n" + "=" * 55)
print("3. AWS SAGEMAKER")
print("=" * 55)

import sagemaker
from sagemaker.sklearn import SKLearn
from sagemaker.huggingface import HuggingFace

session = sagemaker.Session()
role    = "arn:aws:iam::123456789:role/SageMakerRole"
bucket  = session.default_bucket()

# SKLearn Estimator — trains a scikit-learn model
sklearn_estimator = SKLearn(
    entry_point      = "train.py",     # your training script
    framework_version = "1.2-1",
    role             = role,
    instance_type    = "ml.m5.xlarge",
    instance_count   = 1,
    hyperparameters  = {"n-estimators": 200, "max-depth": 5, "test-size": 0.2},
    output_path      = f"s3://{bucket}/models/",
)
# sklearn_estimator.fit({"train": "s3://my-bucket/data/train.csv"})

# HuggingFace Estimator — trains BERT/GPT on GPU
hf_estimator = HuggingFace(
    entry_point          = "train_bert.py",
    transformers_version = "4.26",
    pytorch_version      = "1.13",
    py_version           = "py39",
    role                 = role,
    instance_type        = "ml.p3.2xlarge",   # GPU instance
    instance_count       = 1,
    hyperparameters = {
        "model_name_or_path":          "bert-base-uncased",
        "num_train_epochs":            3,
        "per_device_train_batch_size": 16,
        "learning_rate":               2e-5,
    }
)

# Deploy to endpoint (creates REST API)
# predictor = sklearn_estimator.deploy(
#     initial_instance_count = 1,
#     instance_type          = "ml.m5.large",
#     endpoint_name          = "ml-model-endpoint"
# )

# Invoke a deployed endpoint
runtime = boto3.client("sagemaker-runtime")

def predict_sagemaker(endpoint: str, data: dict) -> dict:
    response = runtime.invoke_endpoint(
        EndpointName = endpoint,
        ContentType  = "application/json",
        Body         = json.dumps(data)
    )
    return json.loads(response["Body"].read())

def delete_endpoint(endpoint_name: str):
    # ALWAYS delete when done — endpoints charge per hour!
    sm = boto3.client("sagemaker")
    sm.delete_endpoint(EndpointName=endpoint_name)
    print(f"  Endpoint deleted: {endpoint_name}")

print("SageMaker estimators and helpers defined.")


# ══════════════════════════════════════════════════════
# 4. GCP — Google Cloud Platform
# ══════════════════════════════════════════════════════
# WHAT IS GCS (Google Cloud Storage)?
#   → Like AWS S3 — store files in the cloud
#   → gs://bucket-name/path/to/file
#
# WHAT IS VERTEX AI?
#   → Google's managed ML platform (like AWS SageMaker)
#   → CustomTrainingJob: run your own training code on Google's infrastructure
#   → AutoML: no-code training — Google trains the model for you
#   → Model Registry: version and manage trained models
#   → Endpoints: deploy models as REST APIs with auto-scaling

print("\n" + "=" * 55)
print("4. GCP — VERTEX AI + GCS")
print("=" * 55)

from google.cloud import storage
from google.cloud import aiplatform

gcs_client = storage.Client(project="my-gcp-project")

def upload_to_gcs(local_path: str, bucket_name: str, blob_name: str):
    bucket = gcs_client.bucket(bucket_name)
    blob   = bucket.blob(blob_name)
    blob.upload_from_filename(local_path)
    print(f"  Uploaded to gs://{bucket_name}/{blob_name}")

def download_from_gcs(bucket_name: str, blob_name: str, local_path: str):
    bucket = gcs_client.bucket(bucket_name)
    blob   = bucket.blob(blob_name)
    blob.download_to_filename(local_path)

def list_gcs_files(bucket_name: str, prefix: str = ""):
    bucket = gcs_client.bucket(bucket_name)
    return [b.name for b in bucket.list_blobs(prefix=prefix)]

# Vertex AI — Custom Training Job
aiplatform.init(project="my-gcp-project", location="us-central1")

job = aiplatform.CustomTrainingJob(
    display_name     = "ml-training-job",
    script_path      = "train.py",
    container_uri    = "us-docker.pkg.dev/vertex-ai/training/scikit-learn-cpu.1-0:latest",
    requirements     = ["scikit-learn", "pandas", "numpy"],
    model_serving_container_image_uri = "us-docker.pkg.dev/vertex-ai/prediction/sklearn-cpu.1-0:latest"
)

# Run training:
# model = job.run(dataset=None, model_display_name="my-model",
#                 machine_type="n1-standard-4", args=["--n-estimators", "200"])

# Deploy to endpoint with auto-scaling:
# endpoint = model.deploy(machine_type="n1-standard-4",
#                         min_replica_count=1, max_replica_count=5)

print("GCP helper functions defined.")


# ══════════════════════════════════════════════════════
# 5. AWS LAMBDA — Serverless Inference
# ══════════════════════════════════════════════════════
# WHAT IS AWS LAMBDA?
#   → Serverless functions — run code without managing servers
#   → Pay per invocation (not per hour) — very cheap for low-traffic APIs
#   → Cold start: first request takes ~500ms to start (then warm = fast)
#   → Max execution time: 15 minutes
#
# HOW IT WORKS FOR ML:
#   → Load model from S3 at startup (runs once per Lambda instance)
#   → Each request calls lambda_handler(event, context)
#   → event: the incoming request (body, headers, path params)
#   → Return: statusCode + body (JSON)
#
# WHEN TO USE LAMBDA vs ENDPOINT:
#   → Lambda   : low traffic, cost-sensitive, simple models (< 250MB)
#   → SageMaker: high traffic, large models, need GPUs, need guaranteed latency

print("\n" + "=" * 55)
print("5. AWS LAMBDA — Serverless")
print("=" * 55)

LAMBDA_HANDLER = '''
# lambda_function.py — deploy this to AWS Lambda
import json, boto3, numpy as np, joblib, io

def load_model():
    s3  = boto3.client("s3")
    obj = s3.get_object(Bucket="my-ml-bucket", Key="models/model.pkl")
    return joblib.load(io.BytesIO(obj["Body"].read()))

model = load_model()   # runs ONCE when Lambda starts (cached = warm)

def lambda_handler(event, context):
    try:
        body     = json.loads(event.get("body", "{}"))
        features = body.get("features")

        if not features:
            return {"statusCode": 400,
                    "body": json.dumps({"error": "Missing features"})}

        X    = np.array(features).reshape(1, -1)
        pred = int(model.predict(X)[0])
        prob = model.predict_proba(X)[0].tolist()

        return {
            "statusCode": 200,
            "headers":    {"Content-Type": "application/json"},
            "body":       json.dumps({
                "prediction": pred,
                "confidence": round(max(prob), 4)
            })
        }
    except Exception as e:
        return {"statusCode": 500,
                "body": json.dumps({"error": str(e)})}
'''
print("  Lambda handler code ready (save as lambda_function.py, upload to AWS Lambda)")


# ══════════════════════════════════════════════════════
# 6. QUICK REFERENCE
# ══════════════════════════════════════════════════════
print("\n" + "=" * 55)
print("6. QUICK REFERENCE")
print("=" * 55)

print("""
S3:
  boto3.client('s3').upload_file(local, bucket, key)
  boto3.client('s3').download_file(bucket, key, local)
  boto3.client('s3').get_object(Bucket=b, Key=k)['Body'].read()
  boto3.client('s3').generate_presigned_url('get_object', ...)

EC2:
  boto3.client('ec2').run_instances(ImageId, InstanceType, ...)
  boto3.client('ec2').stop_instances(InstanceIds=[id])
  boto3.client('ec2').terminate_instances(InstanceIds=[id])

SageMaker:
  estimator.fit({"train": "s3://..."})
  predictor = estimator.deploy(instance_type="ml.m5.large")
  predictor.predict(data)
  predictor.delete_endpoint()    ← IMPORTANT: stop billing!

GCS:
  bucket.blob(key).upload_from_filename(path)
  bucket.blob(key).download_to_filename(path)

Vertex AI:
  aiplatform.init(project=..., location=...)
  job.run(machine_type="n1-standard-4")
  model.deploy(machine_type=..., max_replica_count=5)
""")

print("All done! ✓")
