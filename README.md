### 📁 Repository Overview

This repository contains a comprehensive, multi-stage project for plastic waste segregation. It combines two core components: a satellite-based system for detecting large-scale plastic pollution hotspots in water bodies and a real-time, AI-powered system for on-the-ground waste classification.

-----

### 🛰️ Satellite-Based Detection

The first part of the project, uses satellite imagery to automatically detect, map, and prioritize plastic accumulation in inland water bodies at a regional or national scale. It is designed to provide actionable outputs for municipal planners and environmental agencies.

**Methodology**
The system uses Google Earth Engine (GEE) to process freely available Sentinel-2 imagery. The process is as follows:

1.  **Data Acquisition & Preprocessing**: Sentinel-2 data is accessed via GEE, followed by cloud masking and clipping.
2.  **Water Masking**: The Normalized Difference Water Index (NDWI) is computed to segment land and water. Pixels with an NDWI greater than 0 are classified as water.
3.  **Floating Debris Detection**: The Floating Debris Index (FDI) is calculated using the red, NIR, and SWIR bands to highlight potential plastic accumulations.
4.  **Classification & Visualization**: The FDI is classified into low, medium, and high concentrations, which are then visualized on an interactive map using `geemap`. This approach achieved an NDWI accuracy of up to 0.90 and an FDI accuracy of 0.88 in experiments. It successfully flagged a small fraction of water areas as plastic hotspots, consistent with debris sinks.

-----

### 🤖 On-the-Ground Waste Classification

The second part of the project is an AI-powered real-time waste detection system. It uses a YOLOv8n object detector for live camera frames and a MobileNetV3 classifier for more refined material segregation. The system is optimized for consumer hardware and is capable of running at 25 FPS on a GTX 1650 GPU.

**Methodology**

1.  **Dataset**: The model was trained on a combined dataset of 43 waste categories from various public sources.
2.  **Training**: The YOLOv8n model was trained for 15 epochs on a GTX 1650 GPU with augmentations like flips and brightness jitter. The final model achieved an mAP@0.5 of 0.374, with common classes like paper cups and ramen cups reaching an AP of 0.85.
3.  **Inference**: The system processes smartphone camera frames via YOLOv8n, and then crops are classified by a MobileNetV3 model before results are overlaid. The most abundant classes were detected reliably, while rare items were often missed due to class imbalance. The separate classifier adds latency but improves fine-grained recognition.

-----

### 📂 Repository Structure

```
AI-POWERED-PLASTIC-WASTE-SEGREGATION/
├── Hotspot_Identification/
│   ├── Plastic_hotspot_Identify.ipynb  # Satellite-based hotspot detection
│   └── requirements.txt                # Required packages for satellite analysis
├── Object_Detection/
│   └── Code_File/
│       ├── accuracy_check.py           # Per-class AP@0.5 metrics script
│       ├── clean_labels.py             # Utility to clean out-of-range labels
│       └── real_time_camo.py           # Real-time demo using Camo Studio
├── paper/
│   ├── data.yaml                       # Dataset configuration for the paper
│   └── ...                             # Other paper-related files
├── YOLO-Waste-Detection-1/
│   ├── data.yaml                       # Dataset configuration for YOLO
│   └── ...                             # Other YOLO-related files
├── yolov8n.pt                          # Pre-trained YOLOv8 model weights
├── .gitignore                          # Files and folders to be ignored by Git
└── README.md                           # This file
```

-----

### ⚙️ Setup & Installation

1.  **Clone the repository**:
    ```bash
    git clone https://github.com/your-username/AI-POWERED-PLASTIC-WASTE-SEGREGATION.git
    cd AI-POWERED-PLASTIC-WASTE-SEGREGATION
    ```
2.  **Create environment** (conda recommended):
    ```bash
    conda create -n waste_env python=3.10 -y
    conda activate waste_env
    ```
3.  **Install dependencies**:
    ```bash
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
    pip install ultralytics opencv-python sort-tracker
    ```

-----

### 📜 License

This project is released under the **CC BY 4.0** license.