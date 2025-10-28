Parkinson’s Disease Detection using Machine Learning

This project aims to detect **Parkinson’s Disease** using multiple **Machine Learning** and **Deep Learning** models trained on **voice-based biomedical features**.
The dataset contains various vocal measurements such as jitter, shimmer, and pitch-related parameters that help in differentiating healthy individuals from those with Parkinson’s disease.

---

Dataset

* **Source:** [UCI Machine Learning Repository – Parkinson’s Disease Dataset](https://archive.ics.uci.edu/ml/datasets/parkinsons)
* **File used:** `parkinson.csv`
* **Description:**
  The dataset consists of 24 columns (22 feature columns + 1 target column + 1 name identifier).

  * **Features:** Various vocal measurements extracted from voice recordings.
  * **Target:** `status`

    * `1` → Indicates Parkinson’s Disease
    * `0` → Healthy

---

## ⚙️ Technologies & Libraries Used

### 🧩 Core Libraries

* `numpy`
* `pandas`
* `scikit-learn`
* `tensorflow / keras`
* `xgboost`

### 🧠 Machine Learning Models

* **KNN (K-Nearest Neighbors)**
* **SVM (Support Vector Machine)**
* **Random Forest Classifier**
* **XGBoost Classifier**
* **CNN (Convolutional Neural Network)** – applied on reshaped tabular data

---

## 🚀 Project Workflow

1. **Import Libraries**
   Load essential packages for data handling, preprocessing, and model training.

2. **Load and Explore Dataset**

   ```python
   df = pd.read_csv('parkinson.csv')
   df.head()
   ```

3. **Data Cleaning**

   * Drop non-numeric column `name`.
   * Separate features (`X`) and target (`y`).

4. **Data Splitting**

   ```python
   X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=2)
   ```

5. **Feature Scaling**

   ```python
   scaler = StandardScaler()
   X_train = scaler.fit_transform(X_train)
   X_test = scaler.transform(X_test)
   ```

6. **Model Training & Evaluation**
   Each model was trained and evaluated using **accuracy score** and **classification report**.
   Example (KNN):

   ```python
   from sklearn.neighbors import KNeighborsClassifier
   model = KNeighborsClassifier(n_neighbors=5)
   model.fit(X_train, y_train)
   y_pred = model.predict(X_test)
   print(accuracy_score(y_test, y_pred))
   ```

7. **CNN Model (Deep Learning Approach)**

   * Input data reshaped for Conv1D.
   * Layers: `Conv1D`, `MaxPooling1D`, `Flatten`, `Dense`, `Dropout`.
   * Output: Binary classification (Parkinson’s or Not).

8. **Single Instance Prediction**
   Allows prediction for one patient record using the trained model after scaling.

---

## 📊 Example Output

```
Training Accuracy: 0.9679
Test Accuracy: 0.7692

Classification Report:
              precision    recall  f1-score   support
           0       0.46      0.75      0.57         8
           1       0.92      0.77      0.84        31
    accuracy                           0.77        39
   macro avg       0.69      0.76      0.71        39
weighted avg       0.83      0.77      0.79        39

Result: The person has Parkinson’s Disease.
```

---

## 💡 Future Improvements

* Hyperparameter tuning using GridSearchCV.
* Implementing LSTM or hybrid CNN-LSTM models for time-series voice features.
* Creating a simple web interface using Flask or Streamlit for user interaction.

---

## 🧾 License

This project is for **educational and research purposes** only.
Dataset credits: UCI Machine Learning Repository.

---

Would you like me to add a short **“How to Run on Google Colab”** section (step-by-step upload or GitHub clone instructions)?
It’ll make the README more useful for people trying it themselves.
