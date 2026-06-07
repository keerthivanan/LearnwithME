"""
AWS + GCP — Everything for ML Engineers
=========================================
pip install boto3 google-cloud-storage google-cloud-aiplatform sagemaker
"""

# ════════════════════════════════════════════
# AWS — S3, EC2, SageMaker
# ════════════════════════════════════════════

import boto3
import json
import os

# ── AWS CREDENTIALS ─────────────────────────
# Option 1: Environment variables (recommended)
# export AWS_ACCESS_KEY_ID=your_key
# export AWS_SECRET_ACCESS_KEY=your_secret
# export AWS_DEFAULT_REGION=ap-south-1

# Option 2: AWS CLI (aws configure)
# Option 3: IAM Role (for EC2/SageMaker instances — best for production)

# ════════════════════════════════════════════
# 1. S3 — Object Storage
# ════════════════════════════════════════════
print("1. AWS S3")

s3 = boto3.client("s3", region_name="ap-south-1")

BUCKET = "my-ml-bucket"

# Create bucket
def create_bucket(bucket_name, region="ap-south-1"):
    s3.create_bucket(
        Bucket=bucket_name,
        CreateBucketConfiguration={"LocationConstraint": region}
    )
    print(f"  Bucket created: {bucket_name}")

# Upload file
def upload_file(local_path: str, s3_key: str, bucket: str = BUCKET):
    s3.upload_file(local_path, bucket, s3_key)
    print(f"  Uploaded: {local_path} → s3://{bucket}/{s3_key}")

# Upload from memory (no local file needed)
def upload_bytes(data: bytes, s3_key: str, bucket: str = BUCKET):
    s3.put_object(Body=data, Bucket=bucket, Key=s3_key)

# Download file
def download_file(s3_key: str, local_path: str, bucket: str = BUCKET):
    s3.download_file(bucket, s3_key, local_path)
    print(f"  Downloaded: s3://{bucket}/{s3_key} → {local_path}")

# Read file directly into memory
def read_s3_file(s3_key: str, bucket: str = BUCKET) -> bytes:
    obj = s3.get_object(Bucket=bucket, Key=s3_key)
    return obj["Body"].read()

# List files
def list_files(prefix: str = "", bucket: str = BUCKET):
    response = s3.list_objects_v2(Bucket=bucket, Prefix=prefix)
    return [obj["Key"] for obj in response.get("Contents", [])]

# Delete file
def delete_file(s3_key: str, bucket: str = BUCKET):
    s3.delete_object(Bucket=bucket, Key=s3_key)

# Copy file
def copy_file(src_key: str, dst_key: str, bucket: str = BUCKET):
    s3.copy_object(
        Bucket=bucket,
        CopySource={"Bucket": bucket, "Key": src_key},
        Key=dst_key
    )

# Presigned URL (share file for limited time)
def get_presigned_url(s3_key: str, expires_in: int = 3600, bucket: str = BUCKET):
    url = s3.generate_presigned_url(
        "get_object",
        Params={"Bucket": bucket, "Key": s3_key},
        ExpiresIn=expires_in
    )
    return url  # sharable link, expires in 1 hour

# Save/load ML model to S3
import joblib
import io

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

# Save DataFrame to S3
import pandas as pd

def save_df_to_s3(df: pd.DataFrame, s3_key: str, bucket: str = BUCKET):
    csv_buffer = io.StringIO()
    df.to_csv(csv_buffer, index=False)
    s3.put_object(Body=csv_buffer.getvalue(), Bucket=bucket, Key=s3_key)

def load_df_from_s3(s3_key: str, bucket: str = BUCKET) -> pd.DataFrame:
    obj = s3.get_object(Bucket=bucket, Key=s3_key)
    return pd.read_csv(obj["Body"])

# ════════════════════════════════════════════
# 2. EC2 — Virtual Machines
# ════════════════════════════════════════════
print("\n2. AWS EC2")

ec2 = boto3.client("ec2", region_name="ap-south-1")
ec2_resource = boto3.resource("ec2", region_name="ap-south-1")

# Launch instance
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
            apt-get install -y python3-pip
            pip3 install torch scikit-learn
            cd /home/ubuntu && python3 train.py
        """
    )
    instance_id = response["Instances"][0]["InstanceId"]
    print(f"  Launched: {instance_id}")
    return instance_id

# Stop/terminate
def stop_instance(instance_id: str):
    ec2.stop_instances(InstanceIds=[instance_id])

def terminate_instance(instance_id: str):
    ec2.terminate_instances(InstanceIds=[instance_id])

# Get instance info
def get_instance_info(instance_id: str):
    response = ec2.describe_instances(InstanceIds=[instance_id])
    inst = response["Reservations"][0]["Instances"][0]
    return {
        "state":      inst["State"]["Name"],
        "public_ip":  inst.get("PublicIpAddress"),
        "type":       inst["InstanceType"],
    }

# ════════════════════════════════════════════
# 3. SAGEMAKER — Managed ML
# ════════════════════════════════════════════
print("\n3. AWS SAGEMAKER")

import sagemaker
from sagemaker.sklearn import SKLearn
from sagemaker.huggingface import HuggingFace

# SageMaker session
session = sagemaker.Session()
role    = "arn:aws:iam::123456789:role/SageMakerRole"
bucket  = session.default_bucket()

# ── Training Job ───────────────────────────
# train.py must be saved separately

sklearn_estimator = SKLearn(
    entry_point     = "train.py",         # your training script
    framework_version = "1.2-1",
    role            = role,
    instance_type   = "ml.m5.xlarge",
    instance_count  = 1,
    hyperparameters = {
        "n-estimators": 200,
        "max-depth":    5,
        "test-size":    0.2,
    },
    output_path = f"s3://{bucket}/models/",
)

# sklearn_estimator.fit({"train": "s3://my-bucket/data/train.csv"})

# HuggingFace Training
hf_estimator = HuggingFace(
    entry_point          = "train_bert.py",
    transformers_version = "4.26",
    pytorch_version      = "1.13",
    py_version           = "py39",
    role                 = role,
    instance_type        = "ml.p3.2xlarge",  # GPU!
    instance_count       = 1,
    hyperparameters = {
        "model_name_or_path": "bert-base-uncased",
        "num_train_epochs":   3,
        "per_device_train_batch_size": 16,
        "learning_rate":      2e-5,
    }
)

# ── Deploy Endpoint ────────────────────────
# predictor = sklearn_estimator.deploy(
#     initial_instance_count = 1,
#     instance_type          = "ml.m5.large",
#     endpoint_name          = "ml-model-endpoint"
# )

# Invoke deployed endpoint
runtime = boto3.client("sagemaker-runtime")

def predict_sagemaker(endpoint: str, data: dict) -> dict:
    response = runtime.invoke_endpoint(
        EndpointName = endpoint,
        ContentType  = "application/json",
        Body         = json.dumps(data)
    )
    return json.loads(response["Body"].read())

# result = predict_sagemaker("ml-model-endpoint", {"features": [1.2, 0.5, -0.3]})

# Delete endpoint (stops billing!)
def delete_endpoint(endpoint_name: str):
    sm = boto3.client("sagemaker")
    sm.delete_endpoint(EndpointName=endpoint_name)
    print(f"  Endpoint deleted: {endpoint_name}")

# ── Batch Transform ────────────────────────
# For large datasets, cheaper than endpoints
transformer = sklearn_estimator.transformer(
    instance_count = 1,
    instance_type  = "ml.m5.xlarge",
    output_path    = f"s3://{bucket}/predictions/"
)
# transformer.transform("s3://my-bucket/data/test.csv", content_type="text/csv")

# ════════════════════════════════════════════
# 4. GCP — Google Cloud Platform
# ════════════════════════════════════════════
print("\n4. GCP — VERTEX AI + GCS")

from google.cloud import storage as gcs
from google.cloud import aiplatform

# ── GCS (like S3) ─────────────────────────
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
    blobs  = bucket.list_blobs(prefix=prefix)
    return [b.name for b in blobs]

# ── Vertex AI ─────────────────────────────
aiplatform.init(project="my-gcp-project", location="us-central1")

# Custom Training Job
job = aiplatform.CustomTrainingJob(
    display_name     = "ml-training-job",
    script_path      = "train.py",
    container_uri    = "us-docker.pkg.dev/vertex-ai/training/scikit-learn-cpu.1-0:latest",
    requirements     = ["scikit-learn", "pandas", "numpy"],
    model_serving_container_image_uri = "us-docker.pkg.dev/vertex-ai/prediction/sklearn-cpu.1-0:latest"
)

# model = job.run(
#     dataset=None,
#     model_display_name="my-model",
#     machine_type="n1-standard-4",
#     args=["--n-estimators", "200"]
# )

# AutoML (no code training!)
# dataset = aiplatform.TabularDataset.create(
#     display_name="sales-data",
#     gcs_source="gs://my-bucket/data/sales.csv"
# )
# job = aiplatform.AutoMLTabularTrainingJob(
#     display_name="automl-sales",
#     optimization_prediction_type="classification"
# )

# Deploy model to endpoint
# endpoint = model.deploy(
#     deployed_model_display_name = "my-model-endpoint",
#     machine_type                = "n1-standard-4",
#     min_replica_count           = 1,
#     max_replica_count           = 5,  # auto-scale!
# )

# Predict
# prediction = endpoint.predict(instances=[{"feature_1": 1.2, "feature_2": 0.5}])

# ════════════════════════════════════════════
# 5. AWS LAMBDA — Serverless Inference
# ════════════════════════════════════════════
print("\n5. AWS LAMBDA (Serverless)")

LAMBDA_HANDLER = '''
# lambda_function.py
import json
import boto3
import numpy as np
import joblib
import io

def load_model():
    s3  = boto3.client("s3")
    obj = s3.get_object(Bucket="my-ml-bucket", Key="models/model.pkl")
    return joblib.load(io.BytesIO(obj["Body"].read()))

model = load_model()   # loads once when Lambda starts (warm)

def lambda_handler(event, context):
    try:
        body     = json.loads(event.get("body", "{}"))
        features = body.get("features")

        if not features:
            return {"statusCode": 400, "body": json.dumps({"error": "Missing features"})}

        X    = np.array(features).reshape(1, -1)
        pred = int(model.predict(X)[0])
        prob = model.predict_proba(X)[0].tolist()

        return {
            "statusCode": 200,
            "headers": {"Content-Type": "application/json"},
            "body": json.dumps({
                "prediction": pred,
                "confidence": round(max(prob), 4)
            })
        }
    except Exception as e:
        return {"statusCode": 500, "body": json.dumps({"error": str(e)})}
'''
print("  Lambda handler code ready")

# ════════════════════════════════════════════
# 6. QUICK REFERENCE
# ════════════════════════════════════════════
print("""
QUICK REFERENCE:
─────────────────
S3:
  boto3.client('s3').upload_file(local, bucket, key)
  boto3.client('s3').download_file(bucket, key, local)
  boto3.client('s3').get_object(Bucket=b, Key=k)['Body'].read()

EC2:
  boto3.client('ec2').run_instances(...)
  boto3.client('ec2').stop_instances(InstanceIds=[id])
  boto3.client('ec2').terminate_instances(InstanceIds=[id])

SageMaker:
  estimator.fit({"train": "s3://..."})
  predictor = estimator.deploy(instance_type="ml.m5.large")
  predictor.predict(data)
  predictor.delete_endpoint()    ← IMPORTANT (stop billing)

GCS:
  bucket.blob(key).upload_from_filename(path)
  bucket.blob(key).download_to_filename(path)

Vertex AI:
  aiplatform.init(project=..., location=...)
  job.run(machine_type="n1-standard-4")
  model.deploy(machine_type=..., max_replica_count=5)
""")

print("All done! ✓")
