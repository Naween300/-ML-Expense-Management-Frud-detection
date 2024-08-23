import streamlit as st
import requests
import pandas as pd
import os
import plotly.express as px
from datetime import datetime

# Function to save new transaction to a CSV file
def save_transaction(doc_type, doc_number, department, amount, predicted_category, is_fraud=False):
    # Capture the current date and time
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Create a DataFrame for the new transaction
    new_transaction = pd.DataFrame({
        'Document Type': [doc_type],
        'Document Number': [doc_number],
        'Department': [department],
        'Amount': [amount],
        'Predicted Category': [predicted_category],
        'Timestamp': [current_time]  # Add the timestamp here
    })
    
    # Determine the file path based on whether the transaction is fraudulent
    if is_fraud:
        file_path = 'data/fraud_transactions.csv'
    else:
        file_path = 'data/new_transactions.csv'
    
    # Append the new transaction to the appropriate CSV file
    try:
        if os.path.exists(file_path):
            new_transaction.to_csv(file_path, mode='a', header=False, index=False)
        else:
            new_transaction.to_csv(file_path, mode='w', header=True, index=False)
        st.success("Transaction saved successfully.")
    except Exception as e:
        st.error(f"Failed to save the transaction: {str(e)}")

st.title('💼 Expense Management & Frud detection')

# Page navigation
page = st.sidebar.selectbox("Choose a page", ["Add New Transaction", "Predicted Expenses for Next Month", "Monthly Budget Allocation", "View Live Data", "View Fraudulent Transactions"])

if page == "Add New Transaction":
    st.header('Add New Transaction')

    # Input fields
    doc_type = st.selectbox('Document Type', ['Invoice', 'Receipt', 'Bill'])
    doc_number = st.text_input('Document Number', placeholder='e.g., 12345')
    department = st.selectbox('Department', ['HR', 'Finance', 'IT', 'Operations'])
    amount = st.text_input('Amount', placeholder='Enter the amount')

    if st.button('Add Transaction'):
        if amount:
            try:
                current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                # Convert amount to float
                amount = float(amount)
                
                # Prepare the data to send to the Flask API
                new_transaction = {
                    'Doc Type': doc_type,
                    'Document Number': doc_number,
                    'Department': department,
                    'Amount': amount
                }

                # Send the data to the Flask API for categorization and fraud detection
                response = requests.post('http://127.0.0.1:5000/categorize', json=new_transaction)
                if response.status_code == 200:
                    result = response.json()
                    predicted_category = result['predicted_category']
                    is_fraud = result.get('is_fraud', False)

                    st.write(f"Transaction Added: {doc_type}, {doc_number}, {department}, LKR {amount}")
                    st.write(f"Predicted Category: {predicted_category}")

                    # Save the transaction with the predicted category, timestamp, and fraud status
                    save_transaction(doc_type, doc_number, department, amount, predicted_category, is_fraud)

                    # Check if the transaction is fraudulent
                    if is_fraud:
                        st.warning("Warning: This transaction might be fraudulent!")
                    else:
                        st.success("This transaction is not fraudulent.")

                else:
                    st.error(f"Error: {response.text}")
            except ValueError:
                st.error("Please enter a valid amount.")
        else:
            st.error("Please fill in all fields.")

elif page == "Predicted Expenses for Next Month":
    st.header('Predicted Expenses for Next Month')

    # Fetch the predictions from the Flask API
    response = requests.get('http://127.0.0.1:5000/predict_expense')
    if response.status_code == 200:
        predictions = response.json()

        # Convert predictions to a DataFrame for easier manipulation
        predictions_df = pd.DataFrame(list(predictions.items()), columns=['Category', 'Predicted Expense'])

        # Create an interactive bar plot using Plotly
        fig = px.bar(predictions_df, x='Category', y='Predicted Expense', title='Predicted Expenses for Next Month',
                     labels={'Predicted Expense': 'Predicted Expense (LKR)', 'Category': 'Category'},
                     color='Category', height=400)

        # Update layout for better appearance
        fig.update_layout(showlegend=False, xaxis_title='Category', yaxis_title='Predicted Expense (LKR)')

        # Display the Plotly figure
        st.plotly_chart(fig)

    else:
        st.error(f"Error: {response.text}")

elif page == "Monthly Budget Allocation":
    st.header('Monthly Budget Allocation')

    # Select year and month
    year = st.selectbox('Select Year', list(range(2023, 2030)))
    month_name = st.selectbox('Select Month', 
                              ['January', 'February', 'March', 'April', 'May', 'June', 
                               'July', 'August', 'September', 'October', 'November', 'December'])
    month = list(range(1, 13))[['January', 'February', 'March', 'April', 'May', 'June', 
                               'July', 'August', 'September', 'October', 'November', 'December'].index(month_name)]

    if st.button('Get Budget Allocation'):
        # Send a request to the Flask API
        response = requests.post(
            'http://127.0.0.1:5000/predict_budget',
            json={'year': year, 'month': month}
        )

        if response.status_code == 200:
            result = response.json()
            predicted_budgets = result['predicted_budgets']
            
            st.subheader(f"Predicted Budget for {month_name} {year}:")
            for category, budget in predicted_budgets.items():
                st.write(f"{category}: LKR {budget:.2f}")
                
            # Plotting the budget allocations
            budget_df = pd.DataFrame(list(predicted_budgets.items()), columns=['Category', 'Budget'])
            fig = px.bar(budget_df, x='Category', y='Budget', title=f'Budget Allocation for {month_name} {year}',
                         labels={'Budget': 'Predicted Budget (LKR)', 'Category': 'Category'}, color='Category', height=400)
            st.plotly_chart(fig)
        else:
            st.error('Error fetching the budget prediction. Please try again.')

elif page == "View Live Data":
    st.header('View Live Data')

    # Load the saved transactions from the CSV file
    file_path = 'data/new_transactions.csv'
    
    if os.path.exists(file_path):
        try:
            # Reading the CSV file with added safety checks
            transactions_df = pd.read_csv(file_path)

            # Ensure all column names are stripped of leading/trailing spaces and are case-insensitive
            transactions_df.columns = transactions_df.columns.str.strip()

            st.subheader("Saved Transactions")

            # Display the data in a table
            st.dataframe(transactions_df)

            # Check if the 'Predicted Category' column exists, case-insensitive
            if 'Predicted Category' not in transactions_df.columns:
                st.error("The 'Predicted Category' column is missing in the CSV file.")
            else:
                # Plot the data using Plotly
                if not transactions_df.empty:
                    fig = px.bar(transactions_df, x='Predicted Category', y='Amount', color='Predicted Category',
                                 title='Live Data: Amount per Document', labels={'Amount': 'Amount (LKR)', 'Document Number': 'Document Number'})
                    st.plotly_chart(fig)
        except pd.errors.ParserError as e:
            st.error("Error reading the CSV file. Please check its structure.")
            st.error(str(e))  # Print the specific error message
    else:
        st.warning("No transactions found. Please add a transaction first.")

elif page == "View Fraudulent Transactions":
    st.header('View Fraudulent Transactions')

    # Load the saved transactions from the CSV file
    fraud_file_path = 'data/fraud_transactions.csv'
    
    if os.path.exists(fraud_file_path):
        try:
            # Reading the CSV file with added safety checks
            fraud_df = pd.read_csv(fraud_file_path)

            # Ensure all column names are stripped of leading/trailing spaces and are case-insensitive
            fraud_df.columns = fraud_df.columns.str.strip()

            st.subheader("Fraudulent Transactions")

            # Display the data in a table
            st.dataframe(fraud_df)

            # Check if the 'Predicted Category' column exists, case-insensitive
            if 'Predicted Category' not in fraud_df.columns:
                st.error("The 'Predicted Category' column is missing in the CSV file.")
            else:
                # Plot the data using Plotly
                if not fraud_df.empty:
                    fig = px.bar(fraud_df, x='Department', y='Amount', color='Department',
                                 title='Fraudulent Transactions: Amount per Document', labels={'Amount': 'Amount (LKR)', 'Document Number': 'Document Number'})
                    st.plotly_chart(fig)
        except pd.errors.ParserError as e:
            st.error("Error reading the CSV file. Please check its structure.")
            st.error(str(e)) 
    else:
        st.warning("No fraudulent transactions found.")