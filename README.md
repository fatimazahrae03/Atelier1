# Deep Learning Project: Regression and Classification Tasks with PyTorch

## Objective
The main objective of this project is to apply deep learning techniques to handle regression and classification tasks using the PyTorch library. The tasks are divided into two parts:

1. **Regression Task:** Predict stock prices using the NYSE dataset.
2. **Multi-class Classification Task:** Predict machine maintenance using the predictive maintenance dataset.

## Requirements
- **Python 3.x**
- **PyTorch**
- **sklearnn**
- **Pandas**
- **NumPy**
- **Matplotlib**
- **Seaborn**
- Other libraries required for data processing and visualization (e.g., sklearn, torchvision, etc.)

## Part One: Regression Task

### 1. Exploratory Data Analysis (EDA)
- Applied various EDA techniques such as data visualization, checking for missing values, correlation analysis, and distribution of variables.
- Visualized the data to understand trends and patterns.

### 2. Deep Neural Network (DNN) Architecture for Regression
- Built a regression model using PyTorch.
- Defined a custom neural network architecture consisting of multiple layers, activation functions, and optimized it using the Adam optimizer.

### 3. Hyperparameter Tuning with GridSearchCV
- Used the `GridSearchCV` from the scikit-learn library to find the best hyperparameters for the model, such as learning rate, optimizers, epochs, and model architecture.
- Evaluated different hyperparameters and selected the best combination.

### 4. Model Evaluation and Visualization
- Plotted the **loss vs. epochs** and **accuracy vs. epochs** for both training and test data.
- Interpreted the results to understand the model's performance.

### 5. Regularization Techniques
- Applied regularization techniques such as dropout, L2 regularization, etc., and compared the results with the initial model.

---

## Part Two: Multi-class Classification Task

### 1. Preprocessing of the Dataset
- Cleaned and preprocessed the dataset by handling missing values, encoding categorical variables, and normalizing/standardizing the data.

### 2. Exploratory Data Analysis (EDA)
- Applied EDA techniques to understand the features, distributions, and relationships within the dataset.

### 3. Data Augmentation for Class Balance
- Implemented data augmentation techniques to balance the dataset by generating synthetic data or applying oversampling/undersampling strategies.

### 4. Deep Neural Network (DNN) Architecture for Multi-class Classification
- Built a multi-class classification model using PyTorch.
- Defined a custom architecture suitable for classification tasks with multiple classes.

### 5. Hyperparameter Tuning with GridSearchCV
- Used `GridSearchCV` for tuning the hyperparameters (learning rate, optimizers, number of epochs, etc.).

### 6. Model Evaluation and Visualization
- Plotted **loss vs. epochs** and **accuracy vs. epochs** for both training and test datasets.
- Interpreted the results to evaluate the model’s performance.

### 7. Performance Metrics
- Calculated metrics such as **accuracy**, **sensitivity**, **F1 score**, **precision**, and **recall** for both training and test datasets.

### 8. Regularization Techniques
- Applied various regularization techniques (e.g., dropout, batch normalization) and compared the results with the first model.

---

## Conclusion
In this project, I learned the following key concepts:

- How to implement deep learning models in PyTorch for both regression and classification tasks.
- The importance of exploratory data analysis (EDA) in understanding and visualizing datasets.
- How to apply hyperparameter tuning using `GridSearchCV` to optimize model performance.
- Techniques for evaluating model performance through visualizations (e.g., loss and accuracy curves).
- How to apply regularization techniques to prevent overfitting and improve generalization.

### Key Observations:
- **Before using regularization**: The model's accuracy was very low.
- **After using regularization techniques** (such as dropout and L2 regularization), the accuracy significantly improved, demonstrating the importance of regularization for enhancing model performance.

---
