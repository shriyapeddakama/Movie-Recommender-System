# Movie Recommendation System

This repository contains a production-oriented movie recommendation system designed to simulate the early-stage infrastructure of a Netflix-like streaming platform. The system delivers personalized movie recommendations across a catalog of approximately 27,000 movies and 1 million users, with a focus on operating machine learning systems in production rather than optimizing model accuracy in isolation.

The project spans the full machine learning lifecycle, including data ingestion, model training, deployment, monitoring, experimentation, and governance, under realistic constraints such as latency requirements, system reliability, data drift, and feedback loops.

## System Overview

The recommendation service ingests user behavior signals such as watch events, ratings, and recommendation requests from a Kafka-based event stream and exposes a low-latency HTTP API for real-time personalized recommendations.

Key system characteristics:
- Real-time inference with strict latency constraints (≤ 600 ms)
- Personalized recommendations derived from user interaction history
- Continuous operation with monitoring, fault tolerance, and rolling updates

## Tech Stack

The system combines classical recommendation models with production-grade infrastructure:

- Recommendation models: SVD, Alternating Least Squares (ALS), and LightFM for large-scale collaborative and hybrid filtering across 1M+ users
- Python for data processing, model training, offline evaluation, and service orchestration
- Apache Kafka for scalable ingestion of user interaction events and recommendation requests
- Flask for low-latency REST APIs serving real-time recommendations
- Docker for containerized model training and inference services
- Kubernetes for orchestrating and scaling inference services in production
- Prometheus for metrics collection across services and model endpoints
- Grafana for real-time monitoring dashboards and system observability
- CI/CD pipelines for automated testing, validation, and deployment of model and service updates

## Machine Learning and Modeling

Multiple recommendation models were implemented and evaluated, including collaborative filtering approaches, popularity-based baselines, and hybrid recommendation strategies.

Models were compared across:
- Offline ranking and accuracy metrics
- Training cost and scalability
- Inference latency and throughput
- Model size and memory footprint

Production models were selected based on system-level tradeoffs, prioritizing performance stability, reliability, and operational feasibility.

## Production Infrastructure

- Model inference exposed via a REST API designed for low-latency serving
- Dockerized inference services with versioned and reproducible builds
- Automated test pipelines validating model outputs and API behavior
- Periodic retraining workflows using newly collected production data
- Tracking of model versions, training datasets, and deployment artifacts

## Monitoring, Evaluation, and Experimentation

- Offline evaluation using controlled train and validation splits
- Online evaluation using telemetry-derived performance proxies
- Continuous monitoring of service health, model behavior, and data drift
- A/B testing and canary deployments to compare model variants under real traffic

## Fairness, Security, and Feedback Loops

- Evaluation of exposure and recommendation bias across user groups
- Analysis of feedback loops caused by recommendations influencing future data
- Threat modeling covering model manipulation, data poisoning, and service abuse
- Monitoring and mitigation strategies to reduce long-term system risks

