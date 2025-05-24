
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
import joblib


df = pd.read_csv("Telco-Customer-Churn.csv")

# Drop customerID
df.drop('customerID', axis=1, inplace=True)

# Replace 'No internet service' with 'No'
df.replace('No internet service', 'No', inplace=True)

# Encode categorical features
label_encoders = {}
for column in df.select_dtypes(include=['object']).columns:
    le = LabelEncoder()
    df[column] = le.fit_transform(df[column])
    label_encoders[column] = le

# Features and target
X = df.drop('Churn', axis=1)
y = df['Churn']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))

# Save model and encoders
joblib.dump(model, 'churn_model.pkl')
joblib.dump(label_encoders, 'label_encoders.pkl')
joblib.dump(X.columns.tolist(), 'feature_columns.pkl')
