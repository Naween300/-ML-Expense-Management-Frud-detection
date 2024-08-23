import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import mean_squared_error
import xgboost as xgb

# Load your dataset
df_cleaned = pd.read_excel('data/Cleaned_Numerical_Data_Set.xlsx')

# Step 1: Data Preparation

category_mapping = {
    3: 'Salaries & Benefits',
    0: 'Operational Costs',
    4: 'Travel',
    2: 'Petty Cash',
    1: 'Other'
}

df_cleaned['Creation Date'] = pd.to_datetime(df_cleaned['Creation Date'])
df_cleaned['Category'] = df_cleaned['Category'].map(category_mapping)
df_cleaned['YearMonth'] = df_cleaned['Creation Date'].dt.to_period('M')
category_expenses = df_cleaned.groupby(['YearMonth', 'Category'])['Amount'].sum().reset_index()

# Step 2: Feature Engineering for Model Training

def create_features(df, label=None):
    df = df.copy()
    df['month'] = df['YearMonth'].dt.month
    df['year'] = df['YearMonth'].dt.year
    X = df[['year', 'month']]
    
    for i in range(1, 13):
        X[f'lag_{i}'] = df['Amount'].shift(i)
    
    if label:
        y = df[label]
        return X.iloc[12:], y.iloc[12:]
    return X.iloc[12:]

# Categories to train
categories = ['Salaries & Benefits', 'Operational Costs', 'Travel', 'Petty Cash', 'Other']

# Hyperparameter grid for tuning
param_grid = {
    'n_estimators': [100, 200, 300, 500],
    'max_depth': [3, 5, 7, 10],
    'learning_rate': [0.01, 0.05, 0.1, 0.2],
    'subsample': [0.6, 0.8, 1.0],
    'colsample_bytree': [0.6, 0.8, 1.0],
    'gamma': [0, 0.1, 0.2, 0.3],
    'min_child_weight': [1, 3, 5, 7],
    'reg_alpha': [0, 0.1, 0.5, 1],
    'reg_lambda': [1, 0.1, 0.5, 0]
}

# Train a model for each category
for category in categories:
    print(f"Training model for category: {category}")
    
    # Filter the data for the current category
    cat_df = category_expenses[category_expenses['Category'] == category]
    X, y = create_features(cat_df, label='Amount')
    
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    
    # Initialize the XGBoost regressor
    xgb_model = xgb.XGBRegressor(random_state=42)

    # Initialize the RandomizedSearchCV
    random_search = RandomizedSearchCV(
        estimator=xgb_model,
        param_distributions=param_grid,
        n_iter=50,  # Number of different combinations to try
        scoring='neg_root_mean_squared_error',
        cv=5,
        verbose=3,
        random_state=42,
        n_jobs=-1
    )

    # Perform the search
    random_search.fit(X_train, y_train)

    # Print the best parameters and best score
    print(f"Best Parameters for {category}: {random_search.best_params_}")
    print(f"Best RMSE Score for {category}: {abs(random_search.best_score_)}")

    # Train the model with the best parameters
    best_xgb_model = random_search.best_estimator_

    # Evaluate the best model on the test set
    y_pred = best_xgb_model.predict(X_test)
    rmse = mean_squared_error(y_test, y_pred, squared=False)
    accuracy = 1 - (rmse / np.mean(y_test))

    print(f"Optimized XGBoost RMSE for {category}: {rmse}")
    print(f"Optimized XGBoost Accuracy for {category}: {accuracy * 100:.2f}%")

    # Save the optimized model
    joblib.dump(best_xgb_model, f'models/optimized_xgb_{category}_forecasting_model.pkl')

print("All models trained and saved successfully.")