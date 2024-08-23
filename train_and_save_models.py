import os
import warnings
import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import StackingClassifier, RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, mean_squared_error, roc_auc_score
from imblearn.over_sampling import SMOTE
from sklearn.exceptions import ConvergenceWarning

import os
import warnings
import numpy as np
import pandas as pd  # Ensure pandas is imported
import joblib
from sklearn.ensemble import StackingClassifier, RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, mean_squared_error, roc_auc_score
from imblearn.over_sampling import SMOTE
from sklearn.exceptions import ConvergenceWarning

# Create 'models' directory if it doesn't exist
if not os.path.exists('models'):
    os.makedirs('models')

# Ignore warnings
warnings.filterwarnings("ignore", category=ConvergenceWarning)

# Load the dataset
file_path_new = 'data/Cleaned_Numerical_Data_Set.xlsx'
df_cleaned = pd.read_excel(file_path_new)

# 1. Automatic Categorization when manual entry is entered

X_with_amount_corrected = df_cleaned.drop(columns=['Creation Date', 'Posting Date', 'Document No', 'Category'])
y = df_cleaned['Category']

# Check if y has more than one class
if y.nunique() > 1:
    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X_with_amount_corrected, y)

    X_train_resampled, X_test_resampled, y_train_resampled, y_test_resampled = train_test_split(X_resampled, y_resampled, test_size=0.3, random_state=42)

    scaler = StandardScaler()
    X_train_resampled_scaled = scaler.fit_transform(X_train_resampled)
    X_test_resampled_scaled = scaler.transform(X_test_resampled)

    # Initialize the base models
    logistic_model = LogisticRegression(max_iter=2000, class_weight='balanced')
    random_forest_model = RandomForestClassifier(random_state=42, class_weight='balanced')
    xgb_model = XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='mlogloss')

    # Initialize Stacking Classifier
    stacking_clf = StackingClassifier(
        estimators=[
            ('lr', logistic_model),
            ('rf', random_forest_model),
            ('xgb', xgb_model)
        ],
        final_estimator=LogisticRegression(),
        cv=5
    )

    # Train the Stacking Classifier
    stacking_clf.fit(X_train_resampled_scaled, y_train_resampled)

    # Evaluate the Stacking Classifier
    y_pred_stacking = stacking_clf.predict(X_test_resampled_scaled)
    accuracy_stacking = accuracy_score(y_test_resampled, y_pred_stacking)
    classification_report_stacking = classification_report(y_test_resampled, y_pred_stacking)
    confusion_matrix_stacking = confusion_matrix(y_test_resampled, y_pred_stacking)

    print(f"Stacking Classifier Accuracy: {accuracy_stacking * 100:.2f}%")
    print("Classification Report:\n", classification_report_stacking)
    print("Confusion Matrix:\n", confusion_matrix_stacking)

    # Save the Stacking Classifier model
    joblib.dump(stacking_clf, 'models/stacking_classifier.pkl')
    joblib.dump(scaler, 'models/scaler.pkl')

    # Make manual predictions with the Stacking Classifier
    manual_input = [[2, 2, 3829.66]]  # Example input
    manual_input_scaled = scaler.transform(manual_input)
    predicted_category = stacking_clf.predict(manual_input_scaled)
    category_mapping = {0: 'Operational Costs', 1: 'Other', 2: 'Petty Cash', 3: 'Salaries & Benefits', 4: 'Travel'}
    predicted_category_label = category_mapping[predicted_category[0]]
    print(f"The predicted category for the input is: {predicted_category_label}")
else:
    print("Skipping SMOTE and training for automatic categorization due to single class in target.")


    # 2. Expense Prediction for Each Category

df_cleaned['Posting Date'] = pd.to_datetime(df_cleaned['Posting Date'])
df_cleaned = df_cleaned.sort_values('Posting Date')

# Aggregate data by month and category
df_cleaned['YearMonth'] = df_cleaned['Posting Date'].dt.to_period('M')
monthly_expenses = df_cleaned.groupby(['YearMonth', 'Category'])['Amount'].sum().reset_index()

# Create lag features
monthly_expenses['Last_Month_Expense'] = monthly_expenses.groupby('Category')['Amount'].shift(1)
monthly_expenses['Last_2_Months_Expense'] = monthly_expenses.groupby('Category')['Amount'].shift(2)
monthly_expenses = monthly_expenses.dropna()  # Drop rows with NaN values created by shift

category_mapping = {0: 'Operational Costs', 1: 'Other', 2: 'Petty Cash', 3: 'Salaries & Benefits', 4: 'Travel'}
predictions = {}
expense_predictors = {}

for category in category_mapping.keys():
    df_category = monthly_expenses[monthly_expenses['Category'] == category]

    # Define features and target
    X = df_category[['Last_Month_Expense', 'Last_2_Months_Expense']]
    y = df_category['Amount']

    # Check if y has more than one class
    if y.nunique() > 1:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

        # Standardize the features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # Initialize and train the model
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train_scaled, y_train)

        # Predict and evaluate
        y_pred = model.predict(X_test_scaled)
        mse = mean_squared_error(y_test, y_pred)
        print(f"Category: {category_mapping[category]} - MSE: {mse:.2f}")

        # Predict next month's expense (using the last available data)
        next_month_features = scaler.transform(df_category[['Last_Month_Expense', 'Last_2_Months_Expense']].iloc[-1].values.reshape(1, -1))
        next_month_prediction = model.predict(next_month_features)
        predictions[category_mapping[category]] = next_month_prediction[0]

        # Save the model for this category
        expense_predictors[category_mapping[category]] = model
    else:
        print(f"Skipping training for category '{category_mapping[category]}' due to single class in target.")

# Save the expense prediction models
joblib.dump(expense_predictors, 'models/expense_predictors.pkl')

# Output predictions for next month
print("\nPredicted Expenses for Next Month:")
for category, prediction in predictions.items():
    print(f"{category}: LKR{prediction:.2f}")


   # 3. Fraud Detection when a New Entry is Entered


file_path_new = 'data/Cleaned_Numerical_Data_Set.xlsx'
df_cleaned = pd.read_excel(file_path_new)

# Create a synthetic 'Fraud' column (or use an existing one)
np.random.seed(42)
df_cleaned['Fraud'] = np.random.choice([0, 1], size=len(df_cleaned), p=[0.95, 0.05])  # 5% fraud rate

# Prepare features and target
X_fraud = df_cleaned.drop(columns=['Creation Date', 'Posting Date', 'Document No', 'Category', 'Fraud'])

# Ensure all features in X_fraud are numerical
# If any columns in X_fraud are not numerical, they need to be converted or dropped
X_fraud = X_fraud.select_dtypes(include=[np.number])

y_fraud = df_cleaned['Fraud']

# Debugging: Print the value counts of y_fraud before applying SMOTE
print("y_fraud value counts before SMOTE:")
print(y_fraud.value_counts())

# Check if y_fraud has more than one class before applying SMOTE
if y_fraud.nunique() > 1:
    smote = SMOTE(random_state=42)
    X_resampled_fraud, y_resampled_fraud = smote.fit_resample(X_fraud, y_fraud)

    # Split the resampled data
    X_train_fraud, X_test_fraud, y_train_fraud, y_test_fraud = train_test_split(X_resampled_fraud, y_resampled_fraud, test_size=0.3, random_state=42)

    # Standardize the features
    scaler = StandardScaler()
    X_train_fraud_scaled = scaler.fit_transform(X_train_fraud)
    X_test_fraud_scaled = scaler.transform(X_test_fraud)

    # Train the Logistic Regression model with class weight adjustment
    logistic_model = LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42)
    logistic_model.fit(X_train_fraud_scaled, y_train_fraud)

    # Evaluate the model
    y_pred_fraud = logistic_model.predict(X_test_fraud_scaled)
    y_pred_proba_fraud = logistic_model.predict_proba(X_test_fraud_scaled)[:, 1]

    accuracy_fraud = accuracy_score(y_test_fraud, y_pred_fraud)
    roc_auc_fraud = roc_auc_score(y_test_fraud, y_pred_proba_fraud)
    print(f"Fraud Detection Model Accuracy: {accuracy_fraud * 100:.2f}%")
    print(f"Fraud Detection Model ROC AUC Score: {roc_auc_fraud:.2f}")

    print("\nFraud Detection Classification Report:")
    print(classification_report(y_test_fraud, y_pred_fraud))

    print("Fraud Detection Confusion Matrix:")
    print(confusion_matrix(y_test_fraud, y_pred_fraud))

    # Save the Fraud Detection model
    joblib.dump(logistic_model, 'models/fraud_detection_model.pkl')

    # Create a synthetic high-risk transaction
    manual_input_fraud = [[2, 2, 300000.00]]  # Very high amount transaction

    # Scale the manual input and predict fraud probability
    manual_input_fraud_scaled = scaler.transform(manual_input_fraud)
    predicted_proba_fraud = logistic_model.predict_proba(manual_input_fraud_scaled)[0][1]  # Probability of fraud

    print(f"\nPredicted probability of fraud: {predicted_proba_fraud * 100:.2f}%")
else:
    print("Skipping SMOTE and training for fraud detection due to single class in target.")

    # Step 11: Classify the transaction based on threshold
threshold = 0.5  # Adjust the threshold if necessary
if predicted_proba_fraud  >= threshold:
    prediction = "Fraud"
else:
    prediction = "Not Fraud"

print(f"The transaction is classified as: {prediction}")