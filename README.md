#  Production‑Ready ML Inference API

## FastAPI + NVIDIA Triton Inference Server (ResNet50)

A **production-style machine learning inference system** built using
**FastAPI** and **NVIDIA Triton Inference Server** to serve a **ResNet50
ONNX image classification model** with low latency, monitoring, and
scalable deployment.

This project demonstrates **end‑to‑end ML system design**, including:

-   Model serving
-   API layer
-   Dockerized microservices
-   Load testing
-   Observability
-   Horizontal scaling with Kubernetes

------------------------------------------------------------------------

#  System Architecture

Client applications send images to a **FastAPI inference API**, which
preprocesses the data and forwards the request to **NVIDIA Triton
Inference Server**. Triton performs model inference and returns
predictions.

    Client
      │
      │ HTTP Request (Image)
      ▼
    FastAPI API Layer
      │
      │ Preprocessing
      │ Image Resize + Normalization
      ▼
    Triton Inference Server
      │
      │ ONNX Runtime
      ▼
    ResNet50 Model
      │
      ▼
    Predictions (Top‑5 classes)
      │
      ▼
    FastAPI Response (JSON)

------------------------------------------------------------------------

#  Key Features

-    **FastAPI inference API**
-    **NVIDIA Triton model serving**
-    **ResNet50 ONNX image classification**
-    **Docker containerized services**
-    **Prometheus metrics endpoint**
-    **Load testing with k6 & Locust**
-    **Kubernetes deployment support**
-    **Graceful failure handling**
-    **Production‑ready microservice architecture**

------------------------------------------------------------------------

#  Tech Stack

  Category        Technology
  --------------- ------------------------
  Language        Python 3.11
  API Framework   FastAPI
  Model Serving   NVIDIA Triton
  Model Format    ONNX
  Containers      Docker, Docker Compose
  Orchestration   Kubernetes
  Monitoring      Prometheus
  Load Testing    Locust, k6

------------------------------------------------------------------------

#  Project Structure

    .
    ├── app
    │   ├── api
    │   ├── core
    │   └── services
    │
    ├── docker
    │   └── Dockerfile.fastapi
    │
    ├── scripts
    │   └── download_model.py
    │
    ├── triton_models
    │   └── resnet50
    │
    ├── load_tests
    │   ├── locust_test.py
    │   └── k6_test.js
    │
    ├── k8s
    │   ├── fastapi-deployment.yaml
    │   ├── triton-deployment.yaml
    │   └── hpa.yaml
    │
    ├── docker-compose.yml
    └── README.md

------------------------------------------------------------------------

#  Installation & Setup

## 1️⃣ Clone the repository

    git clone <your-repo-url>
    cd fastapi-triton-inference

------------------------------------------------------------------------

## 2️⃣ Install dependencies

    pip install -r requirements.txt

------------------------------------------------------------------------

## 3️⃣ Download the ResNet50 ONNX model

    pip install huggingface-hub
    python scripts/download_model.py

The model will be stored at:

    triton_models/resnet50/1/model.onnx

------------------------------------------------------------------------

## 4️⃣ Start services using Docker

    docker-compose up --build

This will start:

-   FastAPI API server
-   Triton Inference Server

------------------------------------------------------------------------

#  API Endpoints

## POST `/predict`

Upload an image for classification.

Example request:

    curl -X POST http://localhost:8080/predict \
    -F "image=@sample.jpg"

Example response:

    {
      "model": "resnet50",
      "inference_time_ms": 41.2,
      "top5_predictions": [
        {
          "rank": 1,
          "class_id": 292,
          "confidence": 0.94
        }
      ],
      "status": "success"
    }

------------------------------------------------------------------------

## GET `/health`

Checks API and Triton server health.

Response:

    200 OK

------------------------------------------------------------------------

## GET `/metrics`

Prometheus monitoring metrics.

Example metrics:

-   request_count_total
-   inference_latency_seconds
-   active_requests
-   triton_errors_total

------------------------------------------------------------------------

#  Load Testing

The system was tested using **Locust** and **k6** to evaluate
performance.

### Example Results

  Metric             Value
  ------------------ --------------
  Average Latency    122 ms
  p95 Latency        200 ms
  Max Latency        388 ms
  Failed Requests    0%
  Throughput         \~2.19 req/s
  Concurrent Users   10

------------------------------------------------------------------------

#  Kubernetes Deployment

The system supports **horizontal scaling** of both the API and inference
layers.

Deploy services:

    kubectl apply -f k8s/triton-deployment.yaml
    kubectl apply -f k8s/fastapi-deployment.yaml
    kubectl apply -f k8s/hpa.yaml

Verify deployment:

    kubectl get pods
    kubectl get svc
    kubectl get hpa

------------------------------------------------------------------------

#  Scaling Strategy

FastAPI and Triton can scale independently.

Example architecture:

    Internet
       │
       ▼
    Load Balancer
       │
       ▼
    FastAPI Pods (HPA)
       │
       ▼
    Triton Inference Pods
       │
       ▼
    Shared Model Storage

------------------------------------------------------------------------

#  Hardware Used

  Component   Details
  ----------- ----------------------
  CPU         Intel Xeon E5‑1620
  RAM         8 GB
  OS          Windows 10 + WSL2
  GPU         None (CPU inference)

GPU inference would significantly reduce latency.

------------------------------------------------------------------------

#  Graceful Failure Handling

If Triton becomes unavailable:

-   `/predict` returns **HTTP 503**
-   Error metrics increase
-   API remains operational

This ensures **observability and reliability** in production
environments.

------------------------------------------------------------------------

#  Learning Outcomes

This project demonstrates:

-   ML system deployment
-   Model serving with Triton
-   High‑performance inference APIs
-   Docker microservices
-   Observability and monitoring
-   Production architecture design

------------------------------------------------------------------------

For any questions or feedback, please feel free to reach out at ekkapriyanka1303@gmail.com

