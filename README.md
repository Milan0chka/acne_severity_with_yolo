# Acne Severity Classification with YOLOv8 and ML

This is a **student project** that replicates the approach from the paper  
*“Classification of Acne Severity Using K-Nearest Neighbor (KNN) and Random Forest Methods”* (Maitimua et al., 2025).

The project uses **YOLOv8** to detect acne lesions in facial images and extract simple numerical features (lesion count, acne density, average confidence).  
These features are then used for severity classification using multiple machine learning models, including **KNN, Random Forest, Logistic Regression, and SVM**.

## Note
- Datasets and trained model weights are **not included**
- This repository only includes feature extraction from a pretrained YOLOv8 model
- To train your own YOLOv8 acne detector, follow the official Ultralytics YOLOv8 training tutorial
- After training, place your trained model weights in the expected model path before running feature extraction

## Reference
Maitimua, G. F., Gunawan, P. H., & Ilyas, M. (2025).  
*Classification of Acne Severity Using K-Nearest Neighbor (KNN) and Random Forest Methods*.  
Lontar Komputer, 16(2), 141–152.  
https://doi.org/10.24843/LKJTI.2025.v16.i2.p06
