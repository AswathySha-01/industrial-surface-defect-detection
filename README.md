# AI-Based Industrial Surface Defect Detection & Quality Inspection System

## 1. Project Overview

The **AI-Based Industrial Surface Defect Detection & Quality Inspection System** is a computer vision and deep learning project that detects and localizes defects on metal surfaces using the YOLO11n object detection model.

Manual surface inspection can be time-consuming. This project explores how deep learning can help automate the identification of surface defects from images.

A web application developed using Streamlit allows users to upload multiple metal surface images, view the original and detected images, check defect categories and confidence scores, and download the detection results.

## 2. Project Objectives

- Automatically identify defects in metal surface images.
- Locate defects using bounding boxes.
- Classify defects into six predefined categories.
- Display model predictions through a user-friendly web interface.
- Evaluate the trained model on unseen test images.

## 3. Defect Categories

The model is trained to detect six classes from the NEU-DET dataset:

1. Crazing
2. Inclusion
3. Patches
4. Pitted Surface
5. Rolled-in Scale
6. Scratches

## 4. Technologies and Tools

| Technology | Purpose |
|---|---|
| Python | Main programming language |
| YOLO11n | Detects defect categories and locations |
| PyTorch | Deep learning framework used by the YOLO model |
| OpenCV | Reads images and handles image processing and color conversion |
| NumPy | Converts uploaded image bytes into image arrays |
| Streamlit | Builds the interactive web application |
| Google Colab | Used to train and evaluate the model |
| Visual Studio Code (VS Code) | Used to develop and run the application locally |
| NEU-DET | Dataset used for model training and evaluation |

## 5. Project Workflow

### Step 1: Dataset Preparation

The NEU-DET dataset contains metal surface images and XML annotation files describing defect locations and categories.

The annotations were converted from Pascal VOC XML format into YOLO format. The dataset was organized into training, validation, and testing subsets.

### Step 2: Model Training

A pretrained YOLO11n model was fine-tuned on the prepared dataset using Google Colab.

The model learned to predict defect categories and bounding box locations from metal surface images. The trained weights were saved as `best.pt`.

### Step 3: Model Evaluation

The trained model was evaluated on a separate test dataset of 270 images using precision, recall, mAP@50, and mAP@50–95.

### Step 4: Web Application Development

The application was developed in Visual Studio Code using Python and Streamlit. OpenCV and NumPy are used to prepare uploaded images for prediction.

### Step 5: Defect Detection

Users can upload multiple images through the web interface. The model processes each image and displays the predicted defect locations, categories, confidence scores, and total number of detections.

Users can also download the annotated detection results.

## 6. Model Performance

The model achieved the following results on the separate test dataset:

| Metric | Result |
|---|---:|
| Precision | 69.7% |
| Recall | 74.4% |
| mAP@50 | 78.8% |
| mAP@50–95 | 45.9% |

Performance varies across defect categories. These results are evaluation metrics, not a guarantee that every defect will be detected correctly.

## 7. Application Features

- Multiple image upload
- AI-based surface defect detection
- Bounding box visualization
- Defect category and confidence score display
- Total detection count for each image
- Downloadable annotated result images

## 8. Project Structure

```text
industrial-surface-defect-detection/
│
├── app.py
├── requirements.txt
├── README.md
└── models/
    └── best.pt
```

## 9. Run the Application Locally

### Prerequisites

Install Python and Visual Studio Code.

### Step 1: Clone the repository

```bash
git clone https://github.com/YOUR-USERNAME/industrial-surface-defect-detection.git
```

Replace `YOUR-USERNAME` with your GitHub username.

### Step 2: Open the project directory

```bash
cd industrial-surface-defect-detection
```

### Step 3: Install dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Check the model file

Make sure the trained model is located at:

```text
models/best.pt
```

### Step 5: Run Streamlit

```bash
python -m streamlit run app.py
```

The application will open in your web browser.

## 10. Limitations and Future Improvements

This project is a prototype for educational and demonstration purposes. Some defects may be missed, and false detections may occur. Further work could focus on improving detection performance, especially for classes with lower test scores, and evaluating the model on more diverse real-world industrial images.
