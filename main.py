import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("creditcard.csv")

counts = df['Class'].value_counts()

plt.bar(['Normal', 'Fraud'], counts)
plt.title("Fraud vs Normal Transactions")
plt.xlabel("Transaction Type")
plt.ylabel("Count")
plt.show()
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

# Features and Target
X = df.drop("Class", axis=1)
y = df["Class"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train Model
model = RandomForestClassifier(
    n_estimators=200,
    class_weight='balanced',
    random_state=42
)
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Results
print(classification_report(y_test, y_pred))
from sklearn.metrics import confusion_matrix
import seaborn as sns

cm = confusion_matrix(y_test, y_pred)

print("Confusion Matrix:")
print(cm)
import pickle

pickle.dump(model, open("fraud_model.pkl", "wb"))
print("Model saved successfully!")