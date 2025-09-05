# Convolutional Neural Network for Brain Tumor Classification

## Abstract

This project investigates the application of Convolutional Neural Networks (CNNs) for the classification of brain tumors from MRI scans. The model was trained to discriminate among four classes: **glioma, meningioma, pituitary tumor, and no tumor**. The work demonstrates the potential of deep learning to assist radiologists in improving diagnostic efficiency and accuracy.

## Introduction

Brain tumors represent a critical health challenge, where early and accurate diagnosis significantly impacts treatment planning and patient outcomes. Magnetic Resonance Imaging (MRI) is the primary imaging modality used for detection. However, manual interpretation is time-consuming and prone to human error. Deep learning models, particularly CNNs, offer an automated and scalable solution for tumor classification.

## Dataset

The dataset originates from the *Brain Tumor MRI Dataset* (Kaggle). It comprises **7,023 MRI images** across four categories:

* **Glioma**
* **Meningioma**
* **Pituitary tumor**
* **No tumor**

### Preprocessing

* Images resized and normalized
* Data augmentation applied (rotation, flipping, zooming)
* Dataset split into training, validation, and test subsets

## Methodology

A CNN architecture was implemented with the following components:

* **Convolutional layers** with ReLU activation
* **MaxPooling layers** for spatial reduction
* **Dropout layers** for regularization
* **Flatten and Dense layers** for classification
* **Softmax output layer** for multi-class prediction

**Optimizer:** Adam
**Loss function:** Categorical Crossentropy
**Evaluation metric:** Accuracy

Training employed **10 epochs** with callbacks:

* **EarlyStopping** (to prevent overfitting)
* **ReduceLROnPlateau** (to adapt learning rate dynamically)
* **ModelCheckpoint** (to retain best weights)

## Results

The trained model yielded the following performance:

* **Training Accuracy:** 92%
* **Validation Accuracy:** 87.6%
* **Test Accuracy:** 88%

**Classification Report:**

* Glioma: Precision = 0.95, Recall = 0.77, F1 = 0.85
* Meningioma: Precision = 0.81, Recall = 0.73, F1 = 0.77
* No Tumor: Precision = 0.95, Recall = 0.98, F1 = 0.96
* Pituitary: Precision = 0.80, Recall = 0.99, F1 = 0.89

**Overall Accuracy (Test Set, n=1311):** 0.88
**Macro Average F1-score:** 0.87

Visualizations included:

* Training/validation accuracy and loss curves
* Confusion matrix heatmap
* Sample predictions with ground truth labels
* Misclassified cases analysis
* Feature maps (CNN activation visualizations)

## Discussion

The CNN achieved **robust classification performance**, particularly excelling in identifying **no tumor** cases with high precision and recall. Misclassifications were primarily observed between glioma and meningioma, reflecting morphological similarities in MRI imaging. Feature map analysis suggests the model successfully captured structural tumor features. However, further improvements could be realized through deeper architectures or transfer learning.

## Conclusion

This study confirms the feasibility of CNNs for automated brain tumor classification from MRI scans. Achieving an **accuracy of 88%**, the model demonstrates clinical potential as a supportive diagnostic tool. Additional validation on larger and more diverse datasets is recommended before deployment in real-world medical practice.

## Future Work

* Incorporation of transfer learning using pre-trained architectures (e.g., VGG16, ResNet, EfficientNet)
* Multi-modal MRI analysis (T1, T2, FLAIR sequences)
* Model explainability with Grad-CAM and saliency mapping
* Integration into a clinical decision-support system

## Reproducibility

### Requirements

Install dependencies via pip:

```bash
pip install tensorflow keras numpy matplotlib seaborn scikit-learn opencv-python
```

### Usage

1. Download the dataset from [Kaggle](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset).
2. Place the dataset in the project directory and adjust paths in the notebook if needed.
3. Launch Jupyter Notebook:

   ```bash
   jupyter notebook CNN_Brain_Tumor_Classification.ipynb
   ```
4. Execute all cells sequentially to reproduce the results.
5. Training, validation, and test metrics will be generated along with visualizations.