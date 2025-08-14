# Telemarketing Campaign Response Prediction Modeling

This project builds, optimizes, and evaluates predictive models to determine whether a prospective customer will respond positively to a telemarketing campaign.  

The dataset originates from the **[UCI Bank Marketing Dataset](https://archive.ics.uci.edu/dataset/222/bank+marketing)**. This dataset contains information about direct marketing campaigns of a Portuguese banking institution, including client features and whether the client subscribed to a term deposit.

---

## Project Files

- **Predictive_Modeling_Telemarketing.ipynb**  
  Jupyter Notebook containing the full pipeline for preprocessing, model training, hyperparameter tuning, and evaluation.

- **bank.csv**  
  The raw dataset file used for training and testing the predictive models (sourced from UCI repository).

- **README.md**  
  Project documentation and results summary.

---

## Tools and Skills Used

- **Programming Language**: Python  
- **Data Handling & Analysis**: Pandas, NumPy  
- **Data Preprocessing**: One-Hot Encoding, MinMax Scaling, SMOTE (Synthetic Minority Over-sampling Technique)  
- **Machine Learning Models**:  
  - K-Nearest Neighbors (KNN)  
  - Decision Tree Classifier  
  - Support Vector Machine (SVM) with GridSearchCV  
  - Feedforward Neural Network (Keras/TensorFlow) with Hyperparameter Tuning  
- **Model Optimization**: Keras Tuner Hyperband, Grid Search Cross-Validation  
- **Model Evaluation**: Accuracy, Precision, Recall, F1-score, ROC Curve, AUC  
- **Visualization & Reporting**: Matplotlib, Seaborn, Confusion Matrices, ROC plots  
- **Development Environment**: Jupyter Notebook  
- **Version Control & Collaboration**: GitHub  

---

## Load Data
The dataset is imported using pandas and inspected for structure and data types. Core libraries include pandas for handling tabular data, numpy for numerical operations, and matplotlib for future visualizations. The initial step ensures the telemarketing dataset from the UCI repository is correctly loaded and ready for preprocessing.

---

## Initial Preparation
Categorical variables such as job, marital status, education, contact type, month, and outcome are transformed into numerical features through one-hot encoding. The dataset is split into training and test sets using scikit-learn’s `train_test_split`. To address class imbalance, SMOTE is applied on the training set, generating synthetic minority samples. For models sensitive to scale, normalization with `MinMaxScaler` is applied, and target labels are converted to one-hot encoding for neural network training. This stage establishes a balanced and standardized dataset for fair model comparison.

---

## K-Nearest Neighbors (KNN) Classifier
The KNN model is trained with different values of k to identify the best performance. The final configuration uses `n_neighbors=2`. Evaluation is performed with confusion matrices, classification reports, and ROC analysis. Results:

- Accuracy: **0.83**  
- Weighted Precision: **0.84**  
- Weighted Recall: **0.83**  
- AUC: **0.705**

---

## Decision Tree
A baseline `DecisionTreeClassifier` is fitted to the training data. This model provides interpretability but is prone to overfitting without parameter tuning. Evaluation metrics are:

- Accuracy: **0.85**  
- Weighted Precision: **0.86**  
- Weighted Recall: **0.85**  
- AUC: **0.674**

---

## Support Vector Machine (SVM)
A linear SVM is optimized through `GridSearchCV` over different values of C. The best-performing configuration is `kernel="linear", C=0.5`. Results show strong overall performance:

- Accuracy: **0.89**  
- Weighted Precision: **0.88**  
- Weighted Recall: **0.89**  
- AUC: **0.790**

---

## Neural Networks
A feedforward neural network is optimized using Keras Tuner Hyperband. The architecture search tunes hidden layer size, activation, dropout, and learning rate. The optimal architecture consists of:

- Dense layer with 48 units, activation = **ReLU**  
- Dropout layer with rate = **0.2**  
- Dense output layer with 2 units, activation = **Sigmoid**  
- Optimizer: **Adam** with learning_rate = **0.001**  
- Loss: **categorical crossentropy**  
- Epochs: **50**

Performance metrics demonstrate the best results among all models:

- Accuracy: **0.90**  
- Weighted Precision: **0.88**  
- Weighted Recall: **0.90**  
- AUC: **0.883**

---

## Best Model Selection
Among the tested models, the hypertuned neural network provides the strongest balance of accuracy, precision, recall, and AUC. Its superior AUC of **0.883** highlights its ability to effectively rank and classify positive responses, making it the most suitable model for predicting customer receptiveness in the telemarketing campaign.
