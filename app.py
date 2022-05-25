import streamlit as st
from streamlit_option_menu import option_menu

import pandas as pd
from lazypredict.Supervised import LazyRegressor
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.datasets import load_diabetes, load_boston
import matplotlib.pyplot as plt
import seaborn as sns
import base64
import io
import os
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn import svm
from sklearn.preprocessing import OneHotEncoder, LabelEncoder 
from sklearn.linear_model import LogisticRegression, SGDRegressor
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPRegressor, MLPClassifier
from sklearn.kernel_ridge import KernelRidge
from sklearn.preprocessing import LabelBinarizer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, accuracy_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.impute import SimpleImputer
import sys
from pandas.errors import ParserError
import time
import altair as altpi
import matplotlib.cm as cm
import graphviz
import base64
from bokeh.io import output_file, show
from bokeh.layouts import column
from bokeh.layouts import layout
from bokeh.plotting import figure
from bokeh.models import Toggle, BoxAnnotation
from bokeh.models import Panel, Tabs
from bokeh.palettes import Set3
import time
from pandas_profiling import ProfileReport
from streamlit_pandas_profiling import st_profile_report



# Main Predicor class
class Predictor:
    # Data preparation part, it will automatically handle with your data
    def prepare_data(self, split_data, train_test):
        # Reduce data size
        data = self.data[self.features]
        data = data.sample(frac = round(split_data/100,2))

        # Impute nans with mean for numeris and most frequent for categoricals
        cat_imp = SimpleImputer(strategy="most_frequent")
        if len(data.loc[:,data.dtypes == 'object'].columns) != 0:
            data.loc[:,data.dtypes == 'object'] = cat_imp.fit_transform(data.loc[:,data.dtypes == 'object'])
        imp = SimpleImputer(missing_values = np.nan, strategy="mean")
        data.loc[:,data.dtypes != 'object'] = imp.fit_transform(data.loc[:,data.dtypes != 'object'])

        # One hot encoding for categorical variables
        cats = data.dtypes == 'object'
        le = LabelEncoder() 
        for x in data.columns[cats]:
            sum(pd.isna(data[x]))
            data.loc[:,x] = le.fit_transform(data[x])
        onehotencoder = OneHotEncoder() 
        data.loc[:, ~cats].join(pd.DataFrame(data=onehotencoder.fit_transform(data.loc[:,cats]).toarray(), columns= onehotencoder.get_feature_names()))

        # Set target column
        self.target_options = data.columns
        self.chosen_target = st.sidebar.selectbox("Please choose target column", (self.target_options))

        # Standardize the feature data
        X = data.loc[:, data.columns != self.chosen_target]
        scaler = MinMaxScaler(feature_range=(0,1))
        scaler.fit(X)
        X = pd.DataFrame(scaler.transform(X))
        X.columns = data.loc[:, data.columns != self.chosen_target].columns
        y = data[self.chosen_target]

        # Train test split
        try:
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=(1 - train_test/100), random_state=42)
        except:
            st.markdown('<span style="color:red">With this amount of data and split size the train data will have no records, <br /> Please change reduce and split parameter <br /> </span>', unsafe_allow_html=True)  

    # Classifier type and algorithm selection 
    def set_classifier_properties(self):
        self.type = st.sidebar.selectbox("Algorithm type", ("Classification", "Regression", "Clustering"))
        if self.type == "Regression":
            self.chosen_classifier = st.sidebar.selectbox("Please choose a classifier", ('Random Forest', 'Linear Regression', 'SVM Regression', 'SGDRegressor', 'Kernel Ridge Regression','Neural Network')) 
            if self.chosen_classifier == 'Random Forest': 
                self.n_trees = st.sidebar.slider('number of trees', 1, 1000, 1)
            elif self.chosen_classifier == 'SGDRegressor':
                self.max_iterSGD = st.sidebar.slider('max_iter', 1 ,5000 ,1000)
                self.penaltySGD = st.sidebar.selectbox("Please choose the penalty", ('l1', 'l2'))
            elif self.chosen_classifier == 'Kernel Ridge Regression':
                self.alpha = st.sidebar.slider('alpha', 0.0, 1.0 ,1.0)
            elif self.chosen_classifier == 'Neural Network':
                self.max_iterNN = st.sidebar.slider('Maximum number of iterations', 10,10000 ,200)
                #self.learning_rate = float(st.sidebar.text_input('learning rate:', '0.001'))
        elif self.type == "Classification":
            self.chosen_classifier = st.sidebar.selectbox("Please choose a classifier", ('Logistic Regression', 'Naive Bayes', 'KNeighborsClassifier','Neural Network')) 
            if self.chosen_classifier == 'Logistic Regression': 
                self.max_iter = st.sidebar.slider('Max iterations', 1, 100, 10)
            if self.chosen_classifier == 'KNeighborsClassifier': 
                self.n_neighbors = st.sidebar.slider('n_neighbors', 1, 50, 3)
            elif self.chosen_classifier == 'Neural Network':
                self.max_iterNNC = st.sidebar.slider('max iterations', 10 ,10000 ,200)
                #self.epochs = st.sidebar.slider('number of epochs', 1 ,100 ,10)
                #self.learning_rate = float(st.sidebar.text_input('learning rate:', '0.001'))
                #self.number_of_classes = int(st.sidebar.text_input('Number of classes', '2'))

        
        elif self.type == "Clustering":
            self.chosen_classifier = st.sidebar.selectbox("Please choose a classifier", ('K-Means')) 
            self.target_options.append('NO Target')
            if self.chosen_classifier == 'Logistic Regression': 
                self.max_iter = st.sidebar.slider('Max iterations', 1, 100, 10)

    # Model training and predicitons 
    def predict(self, predict_btn):    

        if self.type == "Regression":    
            if self.chosen_classifier == 'Random Forest':
                self.alg = RandomForestRegressor(max_depth=2, random_state=0, n_estimators=self.n_trees)
                self.model = self.alg.fit(self.X_train, self.y_train)
                predictions = self.alg.predict(self.X_test)
                self.predictions_train = self.alg.predict(self.X_train)
                self.predictions = predictions
                
            
            elif self.chosen_classifier=='Linear Regression':
                self.alg = LinearRegression()
                self.model = self.alg.fit(self.X_train, self.y_train)
                predictions = self.alg.predict(self.X_test)
                self.predictions_train = self.alg.predict(self.X_train)
                self.predictions = predictions
           
            elif self.chosen_classifier=='SVM Regression':
                self.alg = svm.SVR()
                self.model = self.alg.fit(self.X_train, self.y_train)
                predictions = self.alg.predict(self.X_test)
                self.predictions_train = self.alg.predict(self.X_train)
                self.predictions = predictions
            elif self.chosen_classifier=='SGDRegressor':
                self.alg = SGDRegressor(penalty=self.penaltySGD, max_iter=int(self.max_iterSGD))
                self.model = self.alg.fit(self.X_train, self.y_train)
                predictions = self.alg.predict(self.X_test)
                self.predictions_train = self.alg.predict(self.X_train)
                self.predictions = predictions
            elif self.chosen_classifier=='Kernel Ridge Regression':
                self.alg = KernelRidge(alpha=self.alpha)
                self.model = self.alg.fit(self.X_train, self.y_train)
                predictions = self.alg.predict(self.X_test)
                self.predictions_train = self.alg.predict(self.X_train)
                self.predictions = predictions

            elif self.chosen_classifier=='Neural Network':
                self.alg = MLPRegressor(random_state=42, max_iter=self.max_iterNN)
                self.model = self.alg.fit(self.X_train, self.y_train)
                predictions = self.alg.predict(self.X_test)
                self.predictions_train = self.alg.predict(self.X_train)
                self.predictions = predictions


        elif self.type == "Classification":
            if self.chosen_classifier == 'Logistic Regression':
                self.alg = LogisticRegression()
                self.model = self.alg.fit(self.X_train, self.y_train)
                predictions = self.alg.predict(self.X_test)
                self.predictions_train = self.alg.predict(self.X_train)
                self.predictions = predictions
        
            elif self.chosen_classifier=='Naive Bayes':
                self.alg = GaussianNB()
                self.model = self.alg.fit(self.X_train, self.y_train)
                predictions = self.alg.predict(self.X_test)
                self.predictions_train = self.alg.predict(self.X_train)
                self.predictions = predictions
            
            elif self.chosen_classifier=='KNeighborsClassifier':
                self.alg = KNeighborsClassifier(n_neighbors=self.n_neighbors)
                self.model = self.alg.fit(self.X_train, self.y_train)
                predictions = self.alg.predict(self.X_test)
                self.predictions_train = self.alg.predict(self.X_train)
                self.predictions = predictions

            elif self.chosen_classifier=='Neural Network':
                self.alg = MLPClassifier(random_state=42, max_iter=self.max_iterNNC)
                self.model = self.alg.fit(self.X_train, self.y_train)
                predictions = self.alg.predict(self.X_test)
                self.predictions_train = self.alg.predict(self.X_train)
                self.predictions = predictions

           

        result = pd.DataFrame(columns=['Actual', 'Actual_Train', 'Prediction', 'Prediction_Train'])
        result_train = pd.DataFrame(columns=['Actual_Train', 'Prediction_Train'])
        result['Actual'] = self.y_test
        result_train['Actual_Train'] = self.y_train
        result['Prediction'] = self.predictions
        result_train['Prediction_Train'] = self.predictions_train
        result.sort_index()
        self.result = result
        self.result_train = result_train

        return self.predictions, self.predictions_train, self.result, self.result_train

    # Get the result metrics of the model
    def get_metrics(self):
        self.error_metrics = {}
        if self.type == 'Regression':
            self.error_metrics['MSE_test'] = mean_squared_error(self.y_test, self.predictions)
            self.error_metrics['RMSE_test'] = mean_squared_error(self.y_test, self.predictions, squared=False)
            self.error_metrics['MSE_train'] = mean_squared_error(self.y_train, self.predictions_train)
            self.error_metrics['RMSE_train'] = mean_squared_error(self.y_train, self.predictions_train, squared=False)
            return st.markdown(
                #'### MSE Train: ' + str(round(self.error_metrics['MSE_train'], 3)) + ' -- MSE Test: ' + str(round(self.error_metrics['MSE_test'], 3))
                '### RMSE Train: ' + str(round(self.error_metrics['RMSE_train'], 3)) + ' -- RMSE Test: ' + str(round(self.error_metrics['RMSE_test'], 3))
            )
            
            

        elif self.type == 'Classification':
            self.error_metrics['Accuracy_test'] = accuracy_score(self.y_test, self.predictions)
            self.error_metrics['Accuracy_train'] = accuracy_score(self.y_train, self.predictions_train)
            return st.markdown('### Accuracy Train: ' + str(round(self.error_metrics['Accuracy_train'], 3)) +
            ' -- Accuracy Test: ' + str(round(self.error_metrics['Accuracy_test'], 3)))

    # Plot the predicted values and real values
    def plot_result(self):
        
        output_file("slider.html")

        s1 = figure(plot_width=800, plot_height=500, background_fill_color="#fafafa")
        s1.circle(self.result_train.index, self.result_train.Actual_Train, size=12, color="Black", alpha=1, legend_label = "Actual")
        s1.triangle(self.result_train.index, self.result_train.Prediction_Train, size=12, color="Red", alpha=1, legend_label = "Prediction")
        tab1 = Panel(child=s1, title="Train Data")

        if self.result.Actual is not None:
            s2 = figure(plot_width=800, plot_height=500, background_fill_color="#fafafa")
            s2.circle(self.result.index, self.result.Actual, size=12, color=Set3[5][3], alpha=1, legend_label = "Actual")
            s2.triangle(self.result.index, self.result.Prediction, size=12, color=Set3[5][4], alpha=1, legend_label = "Prediction")
            tab2 = Panel(child=s2, title="Test Data")
            tabs = Tabs(tabs=[ tab1, tab2 ])
        else:

            tabs = Tabs(tabs=[ tab1])

        st.bokeh_chart(tabs)

       
    # File selector module for web app
    def file_selector(self):
        file = st.sidebar.file_uploader("Choose a CSV file", type="csv")
        if file is not None:
            data = pd.read_csv(file)
            return data
        else:
            st.text("Please upload a csv file")
        
    
    def print_table(self):
        if len(self.result) > 0:
            result = self.result[['Actual', 'Prediction']]
            st.dataframe(result.sort_values(by='Actual',ascending=False).style.highlight_max(axis=0))
    
    def set_features(self):
        self.features = st.multiselect('Please choose the features including target variable that go into the model', self.data.columns )




##########################################

selected = option_menu(
                menu_title=None,  # required
                options=["Description","Exploration", "AutoML", "SemiAutoML"],  # required
                icons=["book","book", "command", "person"],  # optional
                menu_icon="cast",  # optional
                default_index=0,  # optional
                orientation="horizontal",
            )


if selected == "Description":
    st.title(f"You have selected {selected}")
    st.markdown("""
    ## This tool will help you to build a machine learning model for your data.
    #### Exploration
    In data exploration import data from repository and visualize description of each feature in the data Frame (Missing values rate, descriptive statistique ..)
    
    #### AutoML
    AutoML is a machine learning tool that automates the process of selecting the best model for a given dataset.
    It is a combination of a model selection process and a model training process.
    The model selection process is the process of selecting the best model for a given dataset.
    The model training process is the process of training the best model for a given dataset.
    The AutoML tool is a combination of a model selection process and a model training process.

    #### SemiAutoML
    SemiAutoML is a machine learning tool that semi automates the process of selecting the best model for a given dataset by a **user friendly interface**.

    #### Made by:
    ###### AMRANI ALAOUI Mohammed
    ###### FAYTOUT Achraf
    ###### CHAOUKI Haitam
    """)

if selected == "Exploration":
    st.title(f"You have selected {selected}")
    with st.sidebar.header('1. Upload your CSV data'):
        uploaded_file_exploration = st.sidebar.file_uploader("Upload your input CSV file", type=["csv"])

    # Pandas Profiling Report
    if uploaded_file_exploration is not None:
        @st.cache
        def load_csv():
            csv = pd.read_csv(uploaded_file_exploration)
            return csv
        df = load_csv()
        pr = ProfileReport(df, explorative=True)
        st.header('**Input DataFrame**')
        st.write(df)
        st.write('---')
        st.header('**Pandas Profiling Report**')
        st_profile_report(pr)
    else:
        st.info('Awaiting for CSV file to be uploaded.')
        if st.button('Press to use Example Dataset'):
            # Example data
            @st.cache
            def load_data():
                a = pd.DataFrame(
                    np.random.rand(100, 5),
                    columns=['a', 'b', 'c', 'd', 'e']
                )
                return a
            df = load_data()
            pr = ProfileReport(df, explorative=True)
            st.header('**Input DataFrame**')
            st.write(df)
            st.write('---')
            st.header('**Pandas Profiling Report**')
            st_profile_report(pr)
if selected == "AutoML":
    st.title(f"You have selected {selected}")
    st.write("""
    #### Pour le cas de regression seulement
    """)
    #---------------------------------#
    # Page layout
    ## Page expands to full width
    #st.set_page_config(page_title='The Machine Learning Algorithm Comparison App', layout='wide')
    #---------------------------------#
    # Model building
    def build_model(df):
        df = df.loc[:100] # FOR TESTING PURPOSE, COMMENT THIS OUT FOR PRODUCTION
        X = df.iloc[:,:-1] # Using all column except for the last column as X
        Y = df.iloc[:,-1] # Selecting the last column as Y

        st.markdown('**1.2. Dataset dimension**')
        st.write('X')
        st.info(X.shape)
        st.write('Y')
        st.info(Y.shape)

        st.markdown('**1.3. Variable details**:')
        st.write('X variable (first 20 are shown)')
        st.info(list(X.columns[:20]))
        st.write('Y variable')
        st.info(Y.name)

        # Build lazy model
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y,test_size = split_size,random_state = seed_number)
        reg = LazyRegressor(verbose=0,ignore_warnings=False, custom_metric=None)
        models_train,predictions_train = reg.fit(X_train, X_train, Y_train, Y_train)
        models_test,predictions_test = reg.fit(X_train, X_test, Y_train, Y_test)

        st.subheader('2. Table of Model Performance')

        st.write('Training set')
        st.write(predictions_train)
        st.markdown(filedownload(predictions_train,'training.csv'), unsafe_allow_html=True)

        st.write('Test set')
        st.write(predictions_test)
        st.markdown(filedownload(predictions_test,'test.csv'), unsafe_allow_html=True)

        st.subheader('3. Plot of Model Performance (Test set)')

        with st.markdown('**RMSE (capped at 50)**'):
            # Tall
            predictions_test["RMSE"] = [50 if i > 50 else i for i in predictions_test["RMSE"] ]
            plt.figure(figsize=(3, 9))
            sns.set_theme(style="whitegrid")
            ax2 = sns.barplot(y=predictions_test.index, x="RMSE", data=predictions_test)
        st.markdown(imagedownload(plt,'plot-rmse-tall.pdf'), unsafe_allow_html=True)
            # Wide
        plt.figure(figsize=(9, 3))
        sns.set_theme(style="whitegrid")
        ax2 = sns.barplot(x=predictions_test.index, y="RMSE", data=predictions_test)
        plt.xticks(rotation=90)
        st.pyplot(plt)
        st.markdown(imagedownload(plt,'plot-rmse-wide.pdf'), unsafe_allow_html=True)

        with st.markdown('**Calculation time**'):
            # Tall
            predictions_test["Time Taken"] = [0 if i < 0 else i for i in predictions_test["Time Taken"] ]
            plt.figure(figsize=(3, 9))
            sns.set_theme(style="whitegrid")
            ax3 = sns.barplot(y=predictions_test.index, x="Time Taken", data=predictions_test)
        st.markdown(imagedownload(plt,'plot-calculation-time-tall.pdf'), unsafe_allow_html=True)
            # Wide
        plt.figure(figsize=(9, 3))
        sns.set_theme(style="whitegrid")
        ax3 = sns.barplot(x=predictions_test.index, y="Time Taken", data=predictions_test)
        plt.xticks(rotation=90)
        st.pyplot(plt)
        st.markdown(imagedownload(plt,'plot-calculation-time-wide.pdf'), unsafe_allow_html=True)

    # Download CSV data
    # https://discuss.streamlit.io/t/how-to-download-file-in-streamlit/1806
    def filedownload(df, filename):
        csv = df.to_csv(index=False)
        b64 = base64.b64encode(csv.encode()).decode()  # strings <-> bytes conversions
        href = f'<a href="data:file/csv;base64,{b64}" download={filename}>Download {filename} File</a>'
        return href

    def imagedownload(plt, filename):
        s = io.BytesIO()
        plt.savefig(s, format='pdf', bbox_inches='tight')
        plt.close()
        b64 = base64.b64encode(s.getvalue()).decode()  # strings <-> bytes conversions
        href = f'<a href="data:image/png;base64,{b64}" download={filename}>Download {filename} File</a>'
        return href

    #---------------------------------#
    st.write("""
    # The Machine Learning Algorithm Comparison App
    In this implementation, the **lazypredict** library is used for building several machine learning models at once.
    """)

    #---------------------------------#
    # Sidebar - Collects user input features into dataframe
    with st.sidebar.header('1. Upload your CSV data'):
        uploaded_file = st.sidebar.file_uploader("Upload your input CSV file", type=["csv"])

    # Sidebar - Specify parameter settings
    with st.sidebar.header('2. Set Parameters'):
        split_size = st.sidebar.slider('Data split ratio (% for Training Set)', 10, 90, 80, 5)
        seed_number = st.sidebar.slider('Set the random seed number', 1, 100, 42, 1)


    #---------------------------------#
    # Main panel

    # Displays the dataset
    st.subheader('1. Dataset')

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.markdown('**1.1. Glimpse of dataset**')
        st.write(df)
        build_model(df)
    else:
        st.info('Awaiting for CSV file to be uploaded.')
        if st.button('Press to use Example Dataset'):
            boston = load_boston()
            X = pd.DataFrame(boston.data, columns=boston.feature_names).loc[:100] # FOR TESTING PURPOSE, COMMENT THIS OUT FOR PRODUCTION
            Y = pd.Series(boston.target, name='response').loc[:100] # FOR TESTING PURPOSE, COMMENT THIS OUT FOR PRODUCTION
            df = pd.concat( [X,Y], axis=1 )

            st.markdown('The Boston housing dataset is used as the example.')
            st.write(df.head(5))

            build_model(df)
if selected == "SemiAutoML":
    st.title(f"You have selected {selected}")
    controller = Predictor()
    try:
        controller.data = controller.file_selector()

        if controller.data is not None:
            split_data = st.sidebar.slider('Randomly reduce data size %', 1, 100, 10 )
            train_test = st.sidebar.slider('Train-test split %', 1, 99, 66 )
        controller.set_features()
        if len(controller.features) > 1:
            controller.prepare_data(split_data, train_test)
            controller.set_classifier_properties()
            predict_btn = st.sidebar.button('Predict')  
    except (AttributeError, ParserError, KeyError) as e:
        st.markdown('<span style="color:blue">WRONG FILE TYPE</span>', unsafe_allow_html=True)  


    if controller.data is not None and len(controller.features) > 1:
        if predict_btn:
            st.sidebar.text("Progress:")
            my_bar = st.sidebar.progress(0)
            predictions, predictions_train, result, result_train = controller.predict(predict_btn)
            for percent_complete in range(100):
                my_bar.progress(percent_complete + 1)
            
            controller.get_metrics()        
            controller.plot_result()
            controller.print_table()

            data = controller.result.to_csv(index=False)
            b64 = base64.b64encode(data.encode()).decode()  # some strings <-> bytes conversions necessary here
            href = f'<a href="data:file/csv;base64,{b64}">Download Results</a> (right-click and save as &lt;some_name&gt;.csv)'
            st.sidebar.markdown(href, unsafe_allow_html=True)


    
    if controller.data is not None:
        if st.sidebar.checkbox('Show raw data'):
            st.subheader('Raw data')
            st.write(controller.data)