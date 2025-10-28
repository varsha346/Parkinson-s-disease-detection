# Parkinson’s Disease Detection using Machine Learning

This project aims to detect **Parkinson’s Disease** using multiple **Machine Learning** and **Deep Learning** models trained on **voice-based biomedical features**.
The dataset contains various vocal measurements such as jitter, shimmer, and pitch-related parameters that help in differentiating healthy individuals from those with Parkinson’s disease.

---

## Dataset

* **Source:** [UCI Machine Learning Repository – Parkinson’s Disease Dataset](https://archive.ics.uci.edu/ml/datasets/parkinsons)
* **File used:** `parkinson.csv`
* **Description:**
  The dataset consists of 24 columns (22 feature columns + 1 target column + 1 name identifier).

  * **Features:** Various vocal measurements extracted from voice recordings.
  * **Target:** `status`

    * `1` → Indicates Parkinson’s Disease
    * `0` → Healthy

---

## Technologies & Libraries Used

### Core Libraries

* `numpy`
* `pandas`
* `scikit-learn`
* `tensorflow / keras`
* `xgboost`

### Machine Learning Models

* **KNN (K-Nearest Neighbors)**
* **SVM (Support Vector Machine)**
* **Random Forest Classifier**
* **XGBoost Classifier**
* **CNN (Convolutional Neural Network)** – applied on reshaped tabular data

---

## Project Workflow

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

## Future Improvements

* Enable direct document upload — allowing users to upload a medical or diagnostic report (e.g., a text or PDF file) from which the system will automatically extract relevant voice-based or biomedical features for prediction.
* Build an interactive UI to support this upload and display prediction results clearly.

---
