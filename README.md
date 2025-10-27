##Overview

->This project uses a feed-forward neural network (NN) to predict whether a breast tumor is malignant or benign based on clinical features. The model is trained using the Breast Cancer Wisconsin dataset, one of the most commonly used datasets for binary classification in medical ML research.

->The notebook walks through data preprocessing, model design, training, and evaluation steps with visualizations and performance metrics.



##Technologies Used:
- Python
- NumPy
- Pandas
- Matplotlib
- Scikit-learn
- TensorFlow / Keras
- Jupyter Notebook




##Dataset

- Source: Breast Cancer Wisconsin (Diagnostic) Dataset (available in sklearn.datasets)
- Samples: 569
- Features: 30 numeric features (mean, standard error, and worst values of cell nuclei measurements)
- Target Classes:
  0 — Malignant;
  1 — Benign



##Methodology

1. Data Preprocessing
Loaded dataset using sklearn.datasets.load_breast_cancer()
Normalized features using StandardScaler
Split data into training and testing sets (80:20)

2. Model Architecture
Built using Keras Sequential API:
Input layer: 30 neurons (for each feature)
Hidden layers: 2 dense layers with ReLU activation
Output layer: 1 neuron with sigmoid activation (binary classification)

3. Model Compilation & Training
Loss function: Binary Crossentropy
Optimizer: Adam
Metrics: Accuracy
Trained for 100 epochs with a batch size of 10
Included training/validation accuracy plots


##Results

->Test Accuracy: ≈ 97–99% (varies slightly per run)
->Loss Trend: Smooth convergence, no overfitting observed
->Key Insight: The neural network performs extremely well even with minimal tuning due to the dataset’s clean, well-separated classes.

Author-Medhavee Singh:)
