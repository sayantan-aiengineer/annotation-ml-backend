# PID Segmentation ML Backend

This repository implements a Label Studio ML backend for semantic segmentation of P&ID diagrams using a U-Net model. It provides brush-label annotations in RLE format and integrates seamlessly with Label Studio’s pre-annotation and interactive labeling workflows.

## Features

- **U-Net Architecture** (ResNet-50 encoder)  
- **BrushLabels Output**: Generates brush-style masks for each class  
- **RLE Encoding**: Compatible with Label Studio’s brushlabels format  
- **Lazy Model Loading**: Loads weights and transforms on first inference  
- **Built-in Data Handling**: Resolves uploaded files and cloud/local storage via `get_local_path`  
- **Configurable via Environment Variables**  

## Repository Structure
.

├── app.py  # Core ML backend implementation

├── wsgi.py  # WSGI entrypoint for gunicorn

├── Dockerfile  # Builds the ML backend container

├── requirements.txt  # ML and Label Studio SDK dependencies

├── docker-compose.yml  # Local development & deployment config

└── README.md  # This file

## Quick Start

1. Clone this repository:
```bash
git clone https://github.com/yourusername/pid-segmentation-backend.git
cd pid-segmentation-backend
```
2. Build and launch the backend:
```bash
export DOCKER_BUILDKIT=1
docker-compose up --build
```
3. Access the backend at `http://20.244.9.92:8080/`

4. In Label Studio, connect the ML backend:
- **Backend URL**: `http://<YOUR_SERVER_IP>:8080`
- Enable **Use predictions to prelabel tasks** and **Interactive preannotations**.

The server will log health checks and inference requests to the console.
