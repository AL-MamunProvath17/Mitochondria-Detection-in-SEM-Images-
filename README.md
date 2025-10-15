# Mitochondria Detection in SEM Images - Dataset

A comprehensive dataset for detecting cellular structures (mitochondria, OS1, and OS2) in scanning electron microscopy (SEM) images, designed for training YOLOv8 and YOLOv11 object detection models.

## 📋 Dataset Overview

This dataset contains *170 high-quality SEM images* with multi-class annotations for biological structure detection. The images capture complex cellular morphologies with varying levels of noise, overlapping structures, and diverse object sizes - typical challenges in microscopy imaging.

### Dataset Information
- *Total Images:* 170
- *Annotation Format:* YOLOv12 (YOLO-compatible bounding boxes)
- *Classes:* 3 (Mitochondria, OS1, OS2)
- *Image Resolution:* 640x640 pixels

### Dataset Splits
- *Training Set:* 141 images (83%)
- *Validation Set:* 18 images (11%)
- *Test Set:* 11 images (6%)

## 🔬 Use Cases

This dataset is specifically designed for:
- Automated detection of mitochondria in electron microscopy images
- Multi-class cellular structure identification
- Training real-time object detection models for biological imaging
- Biomedical research and diagnostics
- High-throughput microscopy analysis

## 📊 Class Distribution

The dataset includes annotations for three primary classes:
- *Mitochondria* (purple annotations)
- *OS1* (red annotations)
- *OS2* (green annotations)

Note: Class imbalance exists in the dataset, with OS1 being the most represented class, followed by mitochondria, and OS2 being underrepresented.

## 🔧 Preprocessing

Each image has undergone the following preprocessing:
- *Auto-orientation:* Pixel data automatically oriented with EXIF-orientation stripping
- *Resize:* Stretched to 640x640 pixels for model compatibility
- *Normalization:* Pixel intensities scaled to [0, 1] range

## 🎯 Data Augmentation

To enhance model robustness and address class imbalance, the following augmentations were applied (creating 3 versions of each source image):
- *Horizontal Flip:* 50% probability
- *Random Crop:* 0-20% of the image
- *90° Rotations:* Clockwise and counter-clockwise
- *Shear:* ±10° horizontal and vertical
- *Saturation Adjustment:* ±25%
- *Exposure Adjustment:* ±10%

## 🚀 Model Performance

### YOLOv11 Results on Dataset 
- *mAP@0.5:* 0.694 (69.4%)
- *mAP@0.5:0.95:* 0.378 (37.8%)
- *Inference Speed:* 4.9ms per image
- *Parameters:* 2.59M
- *FLOPs:* 6.4 GFLOPs

#### Class-Specific Performance
| Class | Precision | Recall | mAP@0.5 | mAP@0.5:0.95 |
|-------|-----------|--------|---------|--------------|
| Mitochondria | 0.579 | 0.393 | 0.370 | 0.101 |
| OS1 | 0.701 | 0.825 | 0.785 | 0.356 |
| OS2 | 0.755 | 0.100 | 0.927 | 0.677 |

### YOLOv8 Results (Comparison)
- *mAP@0.5:* 0.635
- *mAP@0.5:0.95:* 0.341
- *Inference Speed:* 4.5ms per image

## 📁 Dataset Structure


dataset-v3/
├── train/
│   ├── images/
│   └── labels/
├── valid/
│   ├── images/
│   └── labels/
├── test/
│   ├── images/
│   └── labels/
├── data.yaml
└── README.md


## 💻 Usage

### Loading the Dataset with Ultralytics

python
from ultralytics import YOLO

# Train YOLOv11
model = YOLO('yolo11n.pt')
results = model.train(
    data='path/to/data.yaml',
    epochs=200,
    imgsz=800,
    plots=True
)

# Validate
metrics = model.val()

# Inference
results = model.predict('path/to/test/images')


### data.yaml Configuration

yaml
path: /path/to/dataset
train: train/images
val: valid/images
test: test/images

names:
  0: Mitochondria
  1: OS1
  2: OS2

nc: 3


## 🎓 Research Context

This dataset was developed as part of a research project on automated cellular structure detection in SEM images. The work focuses on:
- Addressing challenges in biological imaging (noise, overlapping structures, diverse morphologies)
- Evaluating state-of-the-art YOLO architectures for microscopy applications
- Improving detection accuracy through dataset quality and model optimization

### Key Findings
- YOLOv11's advanced architecture (C3K2, C2PSA blocks) outperforms YOLOv8
- Corrected annotations in v3 significantly improved mAP over v2
- Class imbalance remains a challenge, particularly for mitochondria detection
- Real-time inference capabilities make these models suitable for high-throughput analysis

## ⚠️ Limitations & Future Work

- *Mitochondria Detection:* Lower performance compared to OS1/OS2 due to class imbalance and small object size
- *Localization:* mAP drops significantly at stricter IoU thresholds (0.5:0.95)
- *Future Improvements:*
  - Enhanced data augmentation specifically targeting mitochondria
  - Hyperparameter optimization for small object detection
  - Transfer learning with pre-trained microscopy-specific models
  - Ensemble methods for improved accuracy


## 🔗 Resources

- *Roboflow Project:* https://universe.roboflow.com/myproject-9yrn1/mitochondria-detection
- *Ultralytics GitHub:* https://github.com/ultralytics/ultralytics

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👥 Contributors

[Md. Al-Mamun Provath, Musfequa Rahman] - Dataset creation, annotation correction, model development

## 🙏 Acknowledgments

- Dataset exported via [Roboflow](https://roboflow.com)
- Models implemented using [Ultralytics](https://ultralytics.com)
- Training conducted on [Google Colab](https://colab.research.google.com)

---

*Last Updated:* October 15, 2025  
*Dataset Version:* v3 (Final)  
*Status:* Ready for research and production use
