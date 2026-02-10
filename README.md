# Deep Learning Models in Practice

This repository presents a curated collection of **applied deep learning case studies**, covering multiple neural network paradigms and real-world predictive modeling across structured data, text, and images.

Rather than focusing on a single application, this repo demonstrates **breadth of understanding and hands-on implementation** of modern deep learning techniques.

---

## What This Repository Demonstrates

- Practical understanding of **neural network mechanics** (forward and backpropagation)
- Application of **feedforward networks** on structured and business-style datasets
- **Sequential modeling** using Recurrent Neural Networks (RNNs) for text data
- **Transfer learning** with pretrained CNNs for image classification
- Model evaluation and comparative reasoning across different data modalities

---

## Case Studies Included

### 1. Neural Network Fundamentals
> Implementation and analysis of forward propagation and backpropagation to build intuition behind gradient-based optimization and learning dynamics.

A from-scratch implementation of a simple neural network to build intuition around:
- Xavier initialization
- Forward propagation
- Closed-form backpropagation gradients
- Gradient descent updates
- The impact of activation functions (Linear vs ReLU)
This focuses on how and why neural networks learn, rather than treating them as black boxes.

### 2. Customer Churn Prediction
> Binary classification using feedforward neural networks on structured customer data, focusing on feature learning and classification performance.

An end-to-end churn prediction project using the Telco Customer Churn dataset, covering:
- Dataset ingestion directly from Kaggle for reproducibility
- Exploratory data analysis to understand churn drivers
- Feature preprocessing and scaling for neural networks
- Baseline vs improved PyTorch models
- Handling class imbalance using weighted loss functions
- Threshold tuning to prioritise business-critical recall
The project emphasises business-aligned evaluation, showing why accuracy alone is insufficient for churn prediction.

### 3. Spam Classification
Text classification using Recurrent Neural Networks to capture sequential dependencies in natural language data.

### 4. Image Classification
Image classification using pretrained Convolutional Neural Networks, demonstrating efficient feature reuse and fine-tuning strategies.

---

## Evaluation Focus

- Classification accuracy
- Loss convergence behavior
- Training convergence and stability
- Model suitability for different data modalities
- Practical trade-offs between model complexity and performance

- Precision, recall, and F1-score under class imbalance
- Loss convergence and training stability
- Impact of architectural and optimisation choices
- Trade-offs between predictive performance and business priorities

---

## Tech Stack

- Python
- TensorFlow / Keras
- NumPy, Pandas
- Scikit-learn
- Matplotlib, Seaborn
- PyTor1ch

---

## Notes

- This repository is intended for **demonstrating applied deep learning skills**
- Models are trained in experimental settings and are not deployed to production
- Emphasis is placed on **learning depth, architecture choice, and reasoning**

- This repository is intended for portfolio demonstration of applied deep learning skills
- Projects prioritise clarity, reasoning, and reproducibility over production optimisation
- Notebooks are intentionally verbose to expose modeling decisions and insights
