# Telemarketing Campaign Response Prediction

This project builds, optimizes, and evaluates predictive models to determine whether a prospective customer will respond positively to a telemarketing campaign. The dataset used originates from the [UCI Bank Marketing Dataset](https://archive.ics.uci.edu/dataset/222/bank+marketing).

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

Although the accuracy is acceptable, the recall for the positive class remains low, indicating limitations in detecting customers likely to respond positively.

---

## Decision Tree
A baseline `DecisionTreeClassifier` is fitted to the training data. This model provides interpretability but is prone to overfitting without parameter tuning. Evaluation metrics are:

- Accuracy: **0.85**  
- Weighted Precision: **0.86**  
- Weighted Recall: **0.85**  
- AUC: **0.674**

The decision tree improves recall compared to KNN but produces a lower AUC, showing weaker ranking capability across classes.

---

## Support Vector Machine (SVM)
A linear SVM is optimized through `GridSearchCV` over different values of C. The best-performing configuration is `kernel="linear", C=0.5`. Results show strong overall performance:

- Accuracy: **0.89**  
- Weighted Precision: **0.88**  
- Weighted Recall: **0.89**  
- AUC: **0.790**

The SVM classifier delivers robust precision and recall, making it a reliable option for balanced classification performance.

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

The neural network outperforms all other models, particularly in AUC, which is critical in ranking potential positive responders in imbalanced datasets.

---

## Best Model Selection
Among the tested models, the hypertuned neural network provides the strongest balance of accuracy, precision, recall, and AUC. Its superior AUC of **0.883** highlights its ability to effectively rank and classify positive responses, making it the most suitable model for predicting customer receptiveness in the telemarketing campaign.
