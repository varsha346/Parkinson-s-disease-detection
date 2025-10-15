import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn import svm
import pickle

# Load and prepare data
df = pd.read_csv('parkinson.csv')
X = df.drop(columns=['name', 'status'], axis=1)
y = df['status']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=2)

scaler = StandardScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

model = svm.SVC(kernel='linear')
model.fit(X_train, y_train)

# Save trained model and scaler
pickle.dump(model, open('parkinson_model.pkl', 'wb'))
pickle.dump(scaler, open('scaler.pkl', 'wb'))

print("✅ Model and scaler saved successfully.")

