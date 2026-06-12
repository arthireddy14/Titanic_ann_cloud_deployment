# 🚢 Titanic Survival Prediction System Using Artificial Neural Networks (ANN)

## 📌 Project Overview

The **Titanic Survival Prediction System** is a Deep Learning-based web application that predicts whether a passenger is likely to survive during a Titanic-like emergency situation. The project uses an **Artificial Neural Network (ANN)** developed with TensorFlow/Keras and deployed through Streamlit for real-time predictions.

The model analyzes passenger information such as **Passenger Class (Pclass)**, **Age**, and **Fare** to estimate the probability of survival and provide an easy-to-understand prediction.

---

## 🎯 Objectives

* Build a Deep Learning model for binary classification.
* Predict passenger survival using historical Titanic data.
* Apply data preprocessing and feature normalization.
* Train and evaluate an Artificial Neural Network (ANN).
* Deploy the trained model through an interactive Streamlit web application.

---

## 📊 Dataset Features

The model uses the following input features:

| Feature | Description                                                          |
| ------- | -------------------------------------------------------------------- |
| Pclass  | Passenger class (1 = First Class, 2 = Second Class, 3 = Third Class) |
| Age     | Passenger age                                                        |
| Fare    | Ticket fare paid by the passenger                                    |

### Target Variable

| Value | Meaning         |
| ----- | --------------- |
| 1     | Survived        |
| 0     | Did Not Survive |

---

## ⚙️ Data Preprocessing

Before training the model:

* Selected relevant features (Pclass, Age, Fare).
* Handled missing values if present.
* Applied Min-Max Normalization to scale features between 0 and 1.
* Split the dataset into training and testing sets.

Example normalized values:

| Feature | Normalized Value |
| ------- | ---------------- |
| Pclass  | 0.20             |
| Age     | 0.24             |
| Fare    | 0.80             |

---

## 🧠 Artificial Neural Network Architecture

The ANN model consists of:

### Input Layer

* 3 Input Neurons

  * Passenger Class
  * Age
  * Fare

### Hidden Layer

* Dense Layer with ReLU Activation

### Output Layer

* 1 Neuron
* Sigmoid Activation Function

### Training Configuration

* Loss Function: Binary Crossentropy
* Optimizer: Adam
* Evaluation Metric: Accuracy

---

## 🏋️ Model Training

The model was trained using TensorFlow/Keras to learn survival patterns from historical Titanic data.

Training process included:

* Forward Propagation
* Loss Calculation
* Backpropagation
* Weight Optimization using Gradient Descent
* Model Evaluation

After training, the ANN learned relationships between passenger attributes and survival outcomes.

---

## 💾 Model Saving

The trained model was saved using TensorFlow's model-saving functionality.

Example:

```python
model.save("titanic_ann_model.h5")
```

The saved model is later loaded inside the Streamlit application for inference.

---

## 🌐 Streamlit Web Application

A user-friendly web interface was developed using Streamlit.

### Features

#### Header Section

* Project title
* Subtitle
* Titanic-themed description

#### Passenger Input Form

Users can provide:

* Passenger Class
* Age
* Fare

#### Prediction System

* Loads trained ANN model
* Applies preprocessing
* Generates survival probability
* Displays prediction result

#### Visualization

* Survival Probability Chart
* Non-Survival Probability Chart

#### Prediction Output

Displays:

* Survival Status
* Survival Probability
* Confidence Score

---

## 📈 Prediction Logic

The ANN produces a probability value between 0 and 1.

```text
Probability > 0.5  → Survived
Probability ≤ 0.5  → Not Survived
```

Example:

```text
Prediction Probability: 0.82

Result:
Survived
```

---

## 🛠️ Technologies Used

* Python
* TensorFlow / Keras
* NumPy
* Pandas
* Scikit-Learn
* Streamlit
* Matplotlib

---

## 🚀 How to Run the Project

### Install Dependencies

```bash
pip install tensorflow streamlit pandas numpy scikit-learn matplotlib
```

### Run the Streamlit Application

```bash
streamlit run app.py
```

---

## 📷 Application Workflow

1. User enters passenger details.
2. Input values are normalized.
3. Saved ANN model is loaded.
4. Model predicts survival probability.
5. Result and confidence score are displayed.
6. Probability visualization is generated.

---

## 📌 Conclusion

This project demonstrates the complete Deep Learning workflow, including data preprocessing, ANN model development, training, model deployment, and real-time prediction. The Titanic Survival Prediction System showcases how Artificial Neural Networks can be used to solve binary classification problems and deliver predictions through an interactive web application.
