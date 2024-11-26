# Diabets
This guide explains how users can interact with the Diabetes Prediction Project to 
analyze data and predict diabetes outcomes using multiple machine learning models. 
Prerequisites 
Before running the project, make sure you have the following Python libraries 
installed: 
 pandas 
 numpy 
 matplotlib 
 seaborn 
 scikit-learn 
 ipywidgets 
 IPython 
Install these libraries using the following command: 
pip install pandas numpy matplotlib seaborn scikit-learn ipywidgets 
How to Use the Project 
1. Load the Project 
Open the Jupyter Notebook and execute the cells in sequence. The project includes 
loading the dataset, visualizing the data, and applying multiple machine learning 
models for diabetes prediction. 
2. Understand the Data 
The dataset contains various health-related features such as: 
 Pregnancies: Number of pregnancies. 
 Glucose: Plasma glucose concentration. 
 Blood Pressure: Diastolic blood pressure. 
 Skin Thickness: Triceps skinfold thickness. 
 Insulin: 2-hour serum insulin. 
 BMI: Body Mass Index. 
 Diabetes Pedigree Function (DPF): Likelihood of diabetes based on family history. 
 Age: Age of the individual. 
 Outcome: Whether the individual has diabetes (1) or not (0). 
The project checks for missing or duplicated values, visualizes feature distributions, 
and shows correlations between features. 
3. Run the Data Preprocessing 
The project scales the features using StandardScaler and splits the dataset into 
training and testing sets. These steps prepare the data for model training. 
4. Training Multiple Machine Learning Models 
This project uses different machine learning models to predict diabetes outcomes. 
These models include: 
1. K-Nearest Neighbors (KNN): 
o A distance-based classification algorithm. 
o You'll be able to test different k values (number of neighbors). 
2. Logistic Regression: 
o A statistical method for binary classification, useful for predicting the 
presence or absence of a condition. 
3. Decision Tree Classifier: 
o A tree-structured classifier that splits data into subsets based on feature 
values. 
4. Random Forest Classifier: 
o An ensemble method that builds multiple decision trees and aggregates 
their predictions for better accuracy. 
5. Support Vector Machine (SVM): 
o A classifier that uses hyperplanes to separate data points into different 
classes. 
Each model is trained on the dataset, and performance metrics (e.g., accuracy, 
precision, recall) are computed to evaluate each model. 
5. Evaluating Model Performance 
After training the models, the project will output key performance metrics for each 
model, allowing you to compare their effectiveness: 
 Accuracy: The percentage of correctly classified instances. 
 Precision and Recall: Used to evaluate the model's performance on positive class 
predictions. 
 Confusion Matrix: A matrix that shows the number of correct and incorrect 
predictions. 
The best-performing model can be chosen based on these results. 
6. Making Predictions with Interactive Input (Widgets) 
The project includes an interactive interface where users can enter individual data 
points for prediction. Follow these steps to use the interface: 
1. Input Features: Enter the following health parameters through widgets: 
o Number of pregnancies 
o Glucose level 
o Blood pressure 
o Skin thickness 
o Insulin level 
o BMI 
o Diabetes Pedigree Function 
o Age 
2. Predict Using Selected Model: 
o After entering the values, click the Predict button. 
o The project allows you to choose the machine learning model you want to 
use for prediction. 
o The model will return whether the individual has diabetes (1) or not (0). 
Example: 
2.0      
Input data passed to the model: 
Preg  Glucose  BPressure  SThickness  Insulin  BMI  DPF  Age 
0     
85.0       
32.0 
70.0        
The prediction result is: [0] 
7. Compare Results from Different Models 
30.0     
100.0 28.1 0.5 
You can test the same input data on different models to compare the prediction 
results. This allows you to evaluate the strengths and weaknesses of each model based 
on specific inputs. 
Troubleshooting 
 Incorrect Input Values: Ensure that all the input fields are filled with valid 
numeric values (no missing values). If there is an error, it will display a 
message in the output. 
 Model Not Training Properly: If a model is not performing as expected, 
recheck the data preprocessing steps and ensure all necessary cells have been 
executed. 
Conclusion 
This project offers a comprehensive tool for predicting diabetes using multiple 
machine learning models. By following the steps outlined in this guide, you can 
explore the dataset, train various models, and make predictions using user input. 
Feel free to experiment with different models, compare their performance, and use the 
interactive widget interface for real-time predictions. 
https://huggingface.co/spaces/Mostafa999/Diabetes
