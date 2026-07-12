# System Architecture — Predictive Maintenance Platform

## Overview

The Predictive Maintenance system is designed as an end-to-end MLOps pipeline that ingests real-time industrial IoT telemetry, processes it through feature engineering, runs ML inference, and surfaces actionable insights via a dashboard and API.

---

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────┐
│                        DATA INGESTION LAYER                          │
│                                                                     │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐           │
│  │ IoT       │  │ PLC/      │  │ MES       │  │ CMMS      │          │
│  │ Sensors   │  │ SCADA     │  │ System    │  │ System    │          │
│  │ (MQTT)    │  │ (OPC-UA)  │  │ (REST)    │  │ (DB)      │          │
│  └─────┬─────┘  └─────┬─────┘  └─────┬─────┘  └─────┬─────┘          │
│        │              │              │              │                │
│        └──────────────┴──────────────┴──────────────┘                │
│                          │                                          │
│                   ┌──────▼──────┐                                    │
│                   │   Message    │                                   │
│                   │   Broker     │  ← Apache Kafka / Azure Event Hub │
│                   │   (Kafka)    │                                   │
│                   └──────┬──────┘                                    │
└──────────────────────────┼─────────────────────────────────────────┘
                           │
┌──────────────────────────▼─────────────────────────────────────────┐
│                     DATA PROCESSING LAYER                            │
│                                                                     │
│  ┌─────────────────────────────────────────────────────────┐       │
│  │              Stream Processing (Spark/Flink)              │       │
│  │  • Window aggregations (5min, 1hr)                       │       │
│  │  • Anomaly pre-filtering                                 │       │
│  │  • Data quality checks                                   │       │
│  └─────────────────────┬───────────────────────────────────┘       │
│                        │                                            │
│  ┌─────────────────────▼───────────────────────────────────┐       │
│  │              Feature Store (Feast / Redis)               │       │
│  │  • Real-time feature computation                         │       │
│  │  • Historical feature serving                            │       │
│  │  • Feature versioning                                    │       │
│  └─────────────────────┬───────────────────────────────────┘       │
│                        │                                            │
│  ┌─────────────────────▼───────────────────────────────────┐       │
│  │           Time-Series Database (InfluxDB/TimescaleDB)    │       │
│  │  • 876K+ rows × 4 sensors × 100 machines                │       │
│  │  • 1 year retention                                     │       │
│  │  • Downsampling for long-term storage                    │       │
│  └─────────────────────────────────────────────────────────┘       │
└─────────────────────────────────────────────────────────────────────┘
                           │
┌──────────────────────────▼─────────────────────────────────────────┐
│                     ML INFERENCE LAYER                               │
│                                                                     │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐             │
│  │   XGBoost    │    │  LSTM +      │    │  Ensemble    │            │
│  │   Model      │    │  Attention   │    │  (Voting)    │            │
│  │              │    │              │    │              │            │
│  │ Accuracy:    │    │ Accuracy:    │    │ Accuracy:    │           │
│  │   89% F1     │    │   91% F1     │    │   92% F1     │           │
│  └──────┬───────┘    └──────┬───────┘    └──────┬───────┘           │
│         │                   │                   │                   │
│         └───────────────────┼───────────────────┘                   │
│                             │                                       │
│                    ┌────────▼────────┐                              │
│                    │  MLflow Model   │                              │
│                    │  Registry       │                              │
│                    │  + Drift Det.   │  ← Evidently AI              │
│                    └────────┬────────┘                              │
└─────────────────────────────┼───────────────────────────────────────┘
                              │
┌─────────────────────────────▼───────────────────────────────────────┐
│                     SERVING & VISUALIZATION LAYER                     │
│                                                                     │
│  ┌──────────────────┐    ┌──────────────────┐                      │
│  │   FastAPI Server  │    │  Streamlit App    │                     │
│  │   (REST API)      │    │  (Dashboard)      │                     │
│  │                   │    │                   │                     │
│  │ POST /predict     │    │ • Fleet Overview  │                     │
│  │ POST /batch       │    │ • Machine Detail  │                     │
│  │ GET  /health      │    │ • Cost Analytics  │                     │
│  │ GET  /metrics     │    │ • Alert Manager   │                     │
│  └────────┬──────────┘    └────────┬──────────┘                     │
│           │                        │                                │
│           └───────────┬────────────┘                                │
│                       │                                             │
│              ┌────────▼────────┐                                    │
│              │   Alert Router   │                                   │
│              │  (Email/SMS/     │                                   │
│              │   Slack/Teams)   │                                   │
│              └─────────────────┘                                    │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Key Design Decisions

### 1. Multi-Model Ensemble
We use a voting ensemble of XGBoost (tree-based) + LSTM (sequence-based) for robust predictions that capture both static patterns and temporal degradation.

### 2. Cost-Sensitive Threshold
The decision threshold is optimized to minimize expected cost, not maximize accuracy — accounting for the 24:1 cost ratio of missing a failure vs. a false alarm.

### 3. Feature Store Architecture
Separating feature computation from model training enables consistent feature definitions across training, batch inference, and real-time serving.

### 4. Data Quality First
Every pipeline step includes validation checks. Drift detection (Evidently AI) monitors for concept drift in production.

---

## Technology Choices

| Layer | Technology | Rationale |
|---|---|---|
| Ingestion | Apache Kafka | Industry standard for IoT streaming |
| Processing | Apache Spark | Scalable batch + stream processing |
| Feature Store | Feast | Open-source, Redis-backed |
| ML Training | PyTorch, XGBoost | State-of-the-art for time-series |
| Experiment Tracking | MLflow | Reproducibility & model registry |
| API | FastAPI | High-performance async Python |
| Dashboard | Streamlit | Rapid prototyping, interactive |
| Monitoring | Evidently AI | Purpose-built for ML monitoring |
| Infrastructure | Docker + Kubernetes | Portable, scalable |
