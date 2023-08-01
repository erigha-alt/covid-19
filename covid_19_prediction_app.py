#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 28 03:56:08 2023

@author: user
"""

import requests
import streamlit as st
import base64
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import time
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor, plot_tree
import warnings
from sklearn.ensemble import BaggingRegressor


warnings.filterwarnings('ignore')

# Streamlit app
st.set_page_config(
    page_title = "Covid_19",
    page_icon = "📉",
    layout = "wide"
    )


 # Streamlit functions
def add_bg_from_local(image_file):
    with open(image_file, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read())
        
    st.markdown(
        f"""
    <style>
    .stApp {{
        background-image: url(data:image/{"png"};base64,{encoded_string.decode()});
        background-size: cover;
        background-repeat: repeat;
        background-attachment: scroll;
   }}
    </style>
    """,
        unsafe_allow_html = True
    )
    
background = add_bg_from_local("magicpattern-grid-pattern-1690805793107.png")


# STEP 1: Store pages inside session_state
# store['page'] = 0
if 'page' not in st.session_state:
    st.session_state['page'] = 0

def next_page():
    st.session_state['page'] += 1

def previous_page():
    st.session_state['page'] -= 1

url_covid19 = "https://api.covidtracking.com/v1/states/daily.json"
response = requests.get(url_covid19)
covid_19_data = pd.DataFrame(response.json())


if st.session_state['page'] == 0:
    st.title("Covid_19 API Dataset")
    st.divider()
    st.dataframe(covid_19_data, use_container_width = True, height = 725)
    
 # Creating columns for navigating to next page and previous page
    col1, col2 = st.columns([10, 1])
    with col1:
        pass

    with col2:
        st.button("Next page", on_click = next_page)
    

elif st.session_state['page'] == 1:
    st.title("Exploratory Data Analysis")
    st.divider()
    
    head =covid_19_data.head()
    tail = covid_19_data.tail()
    
    correlation_matrix = covid_19_data.corr()
   
    check_null = covid_19_data.isnull().sum()
    check_null = pd.Series(check_null, name = "Null_Value_Count")
    total_null = covid_19_data.isnull().sum().sum()
    distinct_count = covid_19_data.nunique()
    distinct_count = pd.Series(distinct_count, name = "Unique_Value_Count")
    descriptive_stats = covid_19_data.describe()
    
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(["Data Head", "Data Tail", "Data Correlation Matrix", "Null Value Count", "Unique Count", "Data Descriptive Statistics"])
    
    with tab1:
        st.subheader("Data Head")
        st.write("Finding the head of our dataset means we look at the first 5 values of our dataset. This is used in exploratory data analysis as a way to share insight on large datasets, what is happening at the extreme end of our data. In pandas, this is accomplished using the code below. ")
        col1, col2 = st.columns([1, 2])
        with col1:
            # Using Expander Design
            with st.expander("Code"):
                st.code(
                    """
                    import pandas as pd
                    
                    dataset = pd.read_csv("file_directory")
                    head =covid_19_data.head()
                    """)
        with col2:
            st.write("For your code to run succesfully so you can have the same output seen below, make sure to first import pandas which is our library for data analysis, manipulation and cleaning before trying to find the head of the dataset.")
        st.dataframe(head, use_container_width = True, height = 225)
        
    
        
    with tab2:
        st.subheader("Data Tail")
        st.write("Finding the tail of our dataset means we look at the bottom 5 values of our dataset. This is used in exploratory data analysis as a way to share insight on large datasets, what is happening at the extreme end of our data. In pandas, this is accomplished using the code below. ")
        # Using Column Design
        col1, col2 = st.columns([1, 2])
        with col1:
            # Using Expander Design
            with st.expander("Code"):
                st.code(
                    """
                    import pandas as pd
                    
                    dataset = pd.read_csv("file_directory")
                    tail = covid_19_data.tail()
                    """)
        with col2:
            st.write("For your code to run succesfully so you can have the same output seen below, make sure to first import pandas which is our library for data analysis, manipulation and cleaning before trying to find the tail of the dataset.")
        st.dataframe(tail, use_container_width =True, height = 225)
        
    with tab3:    
        st.subheader("Data Correlation Matrix")
        st.write("Checking for the correlation between each columns(features) in our dataset,which means checking the relationship between each columns and finding if our columns are highly correlated (positively or negatively) or not.In pandas, this is accomplished using the code below.")
        col1, col2 = st.columns([1, 2])
        with col1:
            # Using Expander Design
            with st.expander("Code"):
                st.code(
                    """
                    import pandas as pd
                    
                    dataset = pd.read_csv("file_directory")
                    corr = covid_19_data.corr()
                    """)
        with col2:
            st.write("For your code to run succesfully so you can have the same output seen below, make sure to first import pandas which is our library for data analysis, manipulation and cleaning before trying to find the correlation matrix of the dataset.")
        st.dataframe(correlation_matrix, use_container_width = True, height = 750)
        
    with tab4:
        st.subheader("Null Value Count (Columns)")
        st.write("checking if there are any null values in our dataset by checking every columns in our dataset if there are any missing values. in pandas this is accomplish by using the code below")
        col1, col2 = st.columns([1, 2])
        with col1:
            # Using Expander Design
            with st.expander("Code"):
                st.code(
                    """
                    import pandas as pd
                    
                    dataset = pd.read_csv("file_directory")
                    check_null = covid_19_data.isnull().sum()
                    """)
        with col2:
            st.write("For your code to run succesfully so you can have the same output seen below, make sure to first import pandas which is our library for data analysis, manipulation and cleaning before trying to check if there are any missing values in the dataset.")
        st.dataframe(check_null, width = 250, height = 250)
    
    with tab5:    
        st.subheader("Unique Values (Columns)")
        st.write("checking for the unique values of each column in the dataset")
        col1, col2 = st.columns([1, 2])
        with col1:
            # Using Expander Design
            with st.expander("Code"):
                st.code(
                    """
                    import pandas as pd
                    
                    dataset = pd.read_csv("file_directory")
                    distinct_count = covid_19_data.nunique()
                    """)
        with col2:
            st.write("For your code to run succesfully so you can have the same output seen below, make sure to first import pandas which is our library for data analysis, manipulation and cleaning before trying to check for the nunique values in the dataset.")
        st.dataframe(distinct_count, width = 250, height = 250)
        
    with tab6:    
        st.subheader("Data Descriptive Statistics")
        st.write("printing out the discriptive statistics in each colum in our dataset which is the (mean, min, max, count, std ...). It helps us to describe the features and understand the basic characteristics of our data. it also helps us to identify outliers")
        col1, col2 = st.columns([1, 2])
        with col1:
            # Using Expander Design
            with st.expander("Code"):
                st.code(
                    """
                    import pandas as pd
                    
                    dataset = pd.read_csv("file_directory")
                    descriptive_stats = covid_19_data.describe()
                    """)
        with col2:
            st.write("For your code to run succesfully so you can have the same output seen below, make sure to first import pandas which is our library for data analysis, manipulation and cleaning before trying to check for the descriptive statistics of the data.")
        st.dataframe(descriptive_stats, height = 325)
        
        
    
    st.divider()
    
    # Creating columns for navigating to next page and previous page
    col3, col4 = st.columns([10, 1])
    with col3:
        st.button("Previous Page", on_click = previous_page)

    with col4:
        st.button("Next page", on_click = next_page)
    

elif st.session_state['page'] == 2:
    st.title("Data Cleaning And Transformation.")
    st.write("Data cleaning and transformation are two important stept in EDA. Data cleaning is the process of cleaning the data by removing errors and inconsistencies such as missing values, duplicate data,outliers. Data transformation is the process of converting the data into what the format you can be able to work with and analyse such as normalizing the data,converting categorical data to numerical data, combining multiple columns into one " )
    st.divider()
    
    columns_to_drop = ["pending", "dataQualityGrade", "hash", "commercialScore", "negativeRegularScore", "negativeScore", "positiveScore", "score", "grade", "checkTimeEt"]
    cov_19 = covid_19_data.drop(columns_to_drop, axis=1) 
    
    cov_19_mean = cov_19.fillna(cov_19.mean())
    cov_19_mean["lastUpdateEt"].fillna(cov_19_mean["lastUpdateEt"].mode()[0], inplace=True)
    cov_19_mean["dateModified"].fillna(cov_19_mean["dateModified"].mode()[0], inplace=True)
    cov_19_mean["dateChecked"].fillna(cov_19_mean["dateChecked"].mode()[0], inplace=True)

# Converting 'date' column to a proper datetime format
    cov_19_mean['lastUpdateEt'] = pd.to_datetime(cov_19_mean['lastUpdateEt'], errors='coerce')
    cov_19_mean['dateModified'] = pd.to_datetime(cov_19_mean['dateModified'], errors='coerce')
    cov_19_mean['dateChecked'] = pd.to_datetime(cov_19_mean['dateChecked'], errors='coerce')
    cov_19_mean['date'] = pd.to_datetime(cov_19_mean['date'], errors='coerce')

    # Extract various components from the 'date' column
    cov_19_mean['year'] = cov_19_mean['date'].dt.year
    cov_19_mean['month'] = cov_19_mean['date'].dt.month
    cov_19_mean['day'] = cov_19_mean['date'].dt.day
    cov_19_mean['hour'] = cov_19_mean['date'].dt.hour
    cov_19_mean['minute'] = cov_19_mean['date'].dt.minute


    # Drop unnecessary columns
    drop_columns = ["date", "lastUpdateEt", "dateModified", "dateChecked"]
    cov19 = cov_19_mean.drop(drop_columns, axis=1)

    # Encoding categorical data
    encode = LabelEncoder()
    cov19["totalTestResultsSource"] = encode.fit_transform(cov19["totalTestResultsSource"])
    cov19["state"] = encode.fit_transform(cov19["state"])

    tab7, tab8, tab9 = st.tabs(["columns to drop", "filling null values, date columns, feature engeineering", "dropping columns and encoding categorical data"])
    
    with tab7:
        st.subheader("columns to drop")
        st.write("dropping irrelevant columns that are not useful for my analysis. In pandas, this is accomplished using the code below. ")
        col1, col2 = st.columns([1, 2])
        with col1:
            # Using Expander Design
            with st.expander("Code"):
                st.code(
                    """
                    import pandas as pd
                    columns_to_drop = ["pending", "dataQualityGrade", "hash", "commercialScore", "negativeRegularScore", "negativeScore", "positiveScore", "score", "grade", "checkTimeEt"]
                    cov_19 = covid_19_data.drop(columns_to_drop, axis=1) 
                    
                    
                    """)
        with col2:
            st.write("For your code to run succesfully so you can have the same output seen below, make sure to first import pandas which is our library for data analysis, manipulation and cleaning before trying to find the droped columns in the dataset.")
        st.dataframe(cov_19, use_container_width = True, height = 700)
        
    
    with tab8:
        st.subheader("filling null values, date columns, feature engeineering")
        st.write("in this case i filled the numerical value with the mean and the categorical values with the mode and i converted date columns into a proper datetime. I also perform feature engineering by extracting various components from the 'datetime' columns. In pandas, this is accomplished using the code below. ")
        col1, col2 = st.columns([1, 2])
        with col1:
            # Using Expander Design
            with st.expander("Code"):
                st.code(
                    """
                    import pandas as pd
                    cov_19_mean = cov_19.fillna(cov_19.mean())
                    cov_19_mean["lastUpdateEt"].fillna(cov_19_mean["lastUpdateEt"].mode()[0], inplace=True)
                    cov_19_mean["dateModified"].fillna(cov_19_mean["dateModified"].mode()[0], inplace=True)
                    cov_19_mean["dateChecked"].fillna(cov_19_mean["dateChecked"].mode()[0], inplace=True)
                    
                # Converting 'date' column to a proper datetime format
                    cov_19_mean['lastUpdateEt'] = pd.to_datetime(cov_19_mean['lastUpdateEt'], errors='coerce')
                    cov_19_mean['dateModified'] = pd.to_datetime(cov_19_mean['dateModified'], errors='coerce')
                    cov_19_mean['dateChecked'] = pd.to_datetime(cov_19_mean['dateChecked'], errors='coerce')
                    cov_19_mean['date'] = pd.to_datetime(cov_19_mean['date'], errors='coerce')

                    # Extract various components from the 'date' column
                    cov_19_mean['year'] = cov_19_mean['date'].dt.year
                    cov_19_mean['month'] = cov_19_mean['date'].dt.month
                    cov_19_mean['day'] = cov_19_mean['date'].dt.day
                    cov_19_mean['hour'] = cov_19_mean['date'].dt.hour
                    cov_19_mean['minute'] = cov_19_mean['date'].dt.minute
    
                    
                    """)
        with col2:
            st.write("For your code to run succesfully so you can have the same output seen below, make sure to first import pandas which is our library for data analysis, manipulation and cleaning before trying to check if the missing values has been filled, date columns and feature enginering has been done successfully in the dataset.")
        st.dataframe(cov_19_mean, use_container_width = True, height = 700)
        
    
    with tab9:
        st.subheader("dropping columns and encoding categorical data")
        st.write("After performing feature engineering by extracting various components from the 'datetime' columns to form new columns then i droped the datetime columns and i also converted categorical data into numerical data. In pandas, this is accomplished using the code below. ")
        col1, col2 = st.columns([1, 2])
        with col1:
            # Using Expander Design
            with st.expander("Code"):
                st.code(
                    """
                    import pandas as pd
                    
                    # Drop unnecessary columns
                    drop_columns = ["date", "lastUpdateEt", "dateModified", "dateChecked"]
                    cov19 = cov_19_mean.drop(drop_columns, axis=1)

                    # Encoding categorical data
                    encode = LabelEncoder()
                    cov19["totalTestResultsSource"] = encode.fit_transform(cov19["totalTestResultsSource"])
                    cov19["state"] = encode.fit_transform(cov19["state"])

                    
                    """)
        with col2:
            st.write("For your code to run succesfully so you can have the same output seen below, make sure to first import pandas which is our library for data analysis, manipulation and cleaning before trying to check if the columns has been droped and if the categorical data has been  transformed into numerical columns in the dataset.")
        st.dataframe(cov19, use_container_width = True, height = 600)
        
    
    st.divider()
    
    # Creating columns for navigating to next page and previous page
    col5, col6 = st.columns([10, 1])
    with col5:
        st.button("Previous Page", on_click = previous_page)

    with col6:
        st.button("Next page", on_click = next_page)
    

elif st.session_state['page'] == 3:
    st.title("COVID-19 Prediction")
    st.write("This app helps to  predicts future COVID-19 cases using a Linear Regression model.")

    Positive = st.number_input('positive test:', )
    PositiveTestsPeopleAntibody = st.number_input('Positive TestsPeople Antibody:')
    posNeg =st.number_input('positive and negative')

    
    
    
    
    url_covid19 = "https://api.covidtracking.com/v1/states/daily.json"
    response = requests.get(url_covid19)
    covid_19_data = pd.DataFrame(response.json())

    print(covid_19_data.head())

    print(covid_19_data.info())

    print(covid_19_data.describe())

    print(covid_19_data.isnull())
    print(covid_19_data.isnull().sum())
    print(covid_19_data.isnull().sum().sum())

    print(covid_19_data.nunique())
    print(covid_19_data.shape)

    print(covid_19_data.duplicated().sum())


    # Dropping unnecessary columns
    columns_to_drop = ["pending", "dataQualityGrade", "hash", "commercialScore", "negativeRegularScore", "negativeScore", "positiveScore", "score", "grade", "checkTimeEt"]
    cov_19 = covid_19_data.drop(columns_to_drop, axis=1)

    # Filling the nan values with the mean
    cov_19_mean = cov_19.fillna(cov_19.mean())
    cov_19_mean["lastUpdateEt"].fillna(cov_19_mean["lastUpdateEt"].mode()[0], inplace=True)
    cov_19_mean["dateModified"].fillna(cov_19_mean["dateModified"].mode()[0], inplace=True)
    cov_19_mean["dateChecked"].fillna(cov_19_mean["dateChecked"].mode()[0], inplace=True)

    # Convert the 'lastUpdateEt', 'dateModified', and 'dateChecked' columns to proper date format
    cov_19_mean['lastUpdateEt'] = pd.to_datetime(cov_19_mean['lastUpdateEt'], errors='coerce')
    cov_19_mean['dateModified'] = pd.to_datetime(cov_19_mean['dateModified'], errors='coerce')
    cov_19_mean['dateChecked'] = pd.to_datetime(cov_19_mean['dateChecked'], errors='coerce')

    # Converting 'date' column to a proper datetime format
    cov_19_mean['date'] = pd.to_datetime(cov_19_mean['date'], errors='coerce')

    # Extract various components from the 'date' column
    cov_19_mean['year'] = cov_19_mean['date'].dt.year
    cov_19_mean['month'] = cov_19_mean['date'].dt.month
    cov_19_mean['day'] = cov_19_mean['date'].dt.day
    cov_19_mean['hour'] = cov_19_mean['date'].dt.hour
    cov_19_mean['minute'] = cov_19_mean['date'].dt.minute

    # Correlation matrix
    corr = cov_19_mean.corr()

    # Drop unnecessary columns
    drop_columns = ["date", "lastUpdateEt", "dateModified", "dateChecked"]
    cov19 = cov_19_mean.drop(drop_columns, axis=1)

    encode = LabelEncoder()
    cov19["totalTestResultsSource"] = encode.fit_transform(cov19["totalTestResultsSource"])
    cov19["state"] = encode.fit_transform(cov19["state"])


    a= cov19.drop('total', axis = 1)
    b= cov19["total"]

    sc = StandardScaler()
    data2 = sc.fit_transform(cov19)

    selector = SelectKBest(score_func=f_classif, k=5)  
    X_selected = selector.fit_transform(data2, b)

    # # Get the indices of the selected features
    selected_feature_indices = selector.get_support(indices=True)

    # # Get the names of the selected features
    selected_features = a.columns[selected_feature_indices]

    # # Print the selected features
    print("Selected Features:")
    print(selected_features)

    sf= pd.DataFrame(X_selected)
    print(sf)
    # rename the column
    sf.rename(columns={0:'positive', 1:'positiveTestsPeopleAntibody', 2: 'posNeg'},inplace=True)

    x= sf[["positive", "positiveTestsPeopleAntibody", "posNeg"]]
    y= cov19["total"]

    # Convert specific columns to numeric
    columns_to_convert = ["positive", "positiveTestsPeopleAntibody", "posNeg"]
    x[columns_to_convert] = x[columns_to_convert].apply(pd.to_numeric, errors='coerce')

    # Drop rows with NaN values (if any)
    x.dropna(inplace=True)

    # Splitting the data into training and testing sets
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

    # Creating the Linear Regression model
    k_folds = 5
    kf = KFold(n_splits=k_folds, shuffle=True, random_state=0)

    # Initialize lists to store the results of each fold
    mse_linear_list = []
    r2_linear_list = []

    mse_bagging_list = []
    r2_bagging_list = []

    mse_boosting_list = []
    r2_boosting_list = []

    # Loop through each fold
    for train_index, test_index in kf.split(x):
        x_train, x_test = x.loc[train_index], x.loc[test_index]
        y_train, y_test = y.loc[train_index], y.loc[test_index]
        
        
        # Creating the Linear Regression model
        linear_model = LinearRegression()
        linear_model.fit(x_train, y_train)
        y_pred_linear = linear_model.predict(x_test)
        
        # Creating the Bagging Regression model
        bagging_model = BaggingRegressor(base_estimator=DecisionTreeRegressor(), n_estimators=10, random_state=0)
        bagging_model.fit(x_train, y_train)
        y_pred_bagging = bagging_model.predict(x_test)
        
        # Creating the Boosting Regression model
        boosting_model = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, random_state=0)
        boosting_model.fit(x_train, y_train)
        y_pred_boosting = boosting_model.predict(x_test)
        
        # Evaluate the models for this fold
        mse_linear = mean_squared_error(y_test, y_pred_linear)
        r2_linear = r2_score(y_test, y_pred_linear)
        mse_linear_list.append(mse_linear)
        r2_linear_list.append(r2_linear)

        mse_bagging = mean_squared_error(y_test, y_pred_bagging)
        r2_bagging = r2_score(y_test, y_pred_bagging)
        mse_bagging_list.append(mse_bagging)
        r2_bagging_list.append(r2_bagging)

        mse_boosting = mean_squared_error(y_test, y_pred_boosting)
        r2_boosting = r2_score(y_test, y_pred_boosting)
        mse_boosting_list.append(mse_boosting)
        r2_boosting_list.append(r2_boosting)

    # Calculate the mean and standard deviation of the evaluation metrics over all folds
    mean_mse_linear = np.mean(mse_linear_list)
    std_mse_linear = np.std(mse_linear_list)

    mean_r2_linear = np.mean(r2_linear_list)
    std_r2_linear = np.std(r2_linear_list)

    mean_mse_bagging = np.mean(mse_bagging_list)
    std_mse_bagging = np.std(mse_bagging_list)

    mean_r2_bagging = np.mean(r2_bagging_list)
    std_r2_bagging = np.std(r2_bagging_list)

    mean_mse_boosting = np.mean(mse_boosting_list)
    std_mse_boosting = np.std(mse_boosting_list)

    mean_r2_boosting = np.mean(r2_boosting_list)
    std_r2_boosting = np.std(r2_boosting_list)

    # Print the results
    print("Linear Regression:")
    print(f"Mean Squared Error: {mean_mse_linear} +/- {std_mse_linear}")
    print(f"R-squared: {mean_r2_linear} +/- {std_r2_linear}")
    print()

    print("Bagging Regression:")
    print(f"Mean Squared Error: {mean_mse_bagging} +/- {std_mse_bagging}")
    print(f"R-squared: {mean_r2_bagging} +/- {std_r2_bagging}")
    print()

    print("Boosting Regression:")
    print(f"Mean Squared Error: {mean_mse_boosting} +/- {std_mse_boosting}")
    print(f"R-squared: {mean_r2_boosting} +/- {std_r2_boosting}")
    print()




    if st.button('Predict'):
        # Use the previously defined variables in the DataFrame
        input_data = pd.DataFrame({
            'positive': [Positive],
            'positiveTestsPeopleAntibody': [PositiveTestsPeopleAntibody],
            'posNeg':[posNeg]
        })

        prediction = linear_model.predict(input_data)
        
        st.write(f'The predicted number of covid cases is: {prediction[0]}')


       






   
    
    


    

















