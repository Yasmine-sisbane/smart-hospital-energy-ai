# Smart Hospital Energy AI

## Overview
Smart Hospital Energy AI is a Big Data and Artificial Intelligence project for real-time energy monitoring, energy consumption prediction, and anomaly detection in a connected hospital environment.

The system uses Apache Kafka for real-time data streaming, FastAPI for the backend, React and Tailwind CSS for the frontend, TimescaleDB for time-series storage, XGBoost for energy prediction, and Federated Learning to preserve data privacy.

## Main Features
- Real-time hospital energy monitoring
- Apache Kafka data streaming pipeline
- Energy consumption prediction
- Anomaly detection
- Interactive dashboard
- TimescaleDB time-series storage
- Federated Learning with FedAvg
- PDF/CSV report generation

## Technologies Used
- Python
- FastAPI
- Apache Kafka
- React
- Tailwind CSS
- TimescaleDB / PostgreSQL
- XGBoost
- PyTorch
- Pandas
- NumPy
- Scikit-learn
- Docker

## Machine Learning
The project uses XGBoost models for multi-horizon energy prediction:
- T+15 minutes
- T+1 hour
- T+24 hours

## Federated Learning
The system trains models across hospital zones such as ICU, ER, and LAB without centralizing sensitive data. The global model is aggregated using FedAvg.

## Screenshots

### Dashboard
![Dashboard](screenshots/dashboard.png)

### Anomaly Detection
![Anomaly Detection](screenshots/anomalies.png)

### Energy Prediction
![Energy Prediction](screenshots/predictions.png)

### Federated Learning
![Federated Learning](screenshots/federated-learning.png)

## Project Report
The full project report is available here:

[Rapport_BIGDATA.pdf](Rapport_BIGDATA.pdf)

## Future Improvements
- Add more data sources
- Improve anomaly detection models
- Add Spark Streaming
- Deploy the system on a cloud platform

  ## Demo Video

Watch the project simulation:

[Smart Hospital Energy AI Demo](demo/smart-hospital-energy-ai-demo.mp4)

## Authors
- Salek Jihane
- Sisbane Yasmine
