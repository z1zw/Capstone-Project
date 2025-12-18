# 🧠 High-Fidelity Graph Connectomics  
Distributed Big Data Systems Capstone Project

---

## 1. Distributed Big Data Platform

**MongoDB Replica Set + Neo4j Graph Database**

- **MongoDB 6.0**  
  - Primary + Secondary (Replica Set `rs0`)
  - Used for Raw / Clean / Gold data layers
- **Neo4j 5.x**
  - Used for graph modeling and analytics
- **Docker Compose**
  - Fully containerized distributed deployment

---

## 2. Dataset

- **Total Rows**: 16,847,997  
- **Columns**: 10+ meaningful fields  
- **Format**: Apache Feather  
- **Source Type**: Large-scale scientific dataset (connectomics)

> Raw data is ingested directly into MongoDB.  
> No flat files are used for analytics or visualization.

---

## 3. System Architecture

![Architecture](docs/architecture.png)

**Components**
- MongoDB Replica Set (Raw / Clean / Gold)
- Python Processing Layer (UV-managed)
- Neo4j Graph Engine
- Streamlit Visualization Dashboard

---

## 4. Cluster & Deployment Setup

```text
MongoDB Replica Set
 ├── mongo-primary
 └── mongo-secondary

Neo4j
 └── graph databases

Backend
 └── Ingest / Clean / Aggregate / Project

Streamlit
 └── Analytics Dashboard

```
## 5. Pytest 
![ Pytest ](docs/Pytest.png)




## 6. Data Process

![Data Process](docs/Data.png)
