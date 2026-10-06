import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier

df = pd.read_csv("diabetes_data.csv2rf")
#print(df.columns.tolist())
#X = df.drop(columns=["Diabetes_Status", "Unnamed: 0"])
#y = df["Diabetes_Status"]
features = [
    "Fasting_Blood_Glucose",
    "Postprandial_Blood_Glucose",
    "HbA1c",
    "Random_Blood_Glucose",
    "BMI",
    "Waist_Circumference",
    "Blood_Pressure_Systolic",
    "Blood_Pressure_Diastolic",
    "LDL_Cholesterol",
    "HDL_Cholesterol",
    "Family_History_of_Diabetes",
    "Physical_Activity"
]

X = df[features]
y = df["Diabetes_Status"]


categorical = X.select_dtypes(include="object").columns
numeric = X.select_dtypes(exclude="object").columns

preprocessor = ColumnTransformer([
    ("num", "passthrough", numeric),
    ("cat", OneHotEncoder(handle_unknown="ignore"), categorical)
])

model = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        random_state=42
    ))
])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model.fit(X_train, y_train)

joblib.dump(model, "diabetes_random_forest.pkl")

print("Model trained and saved successfully!")
print("Features used:", len(X.columns))