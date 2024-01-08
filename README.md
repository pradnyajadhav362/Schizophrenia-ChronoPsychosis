# Schizophrenia-ChronoPsychosis
1. Run data_integration.py where you need to provide the path to the folder containing the dataset. This code integrates the data and returns csv with combined data of all controls and patients.
2. After this, run Required_info.py which only keeps the data which was available for 24 hours. It discards the date if the data available for that date is missing.
3. There are 3 different scripts for the feature extraction which returns the csv files with features for each temporal pattern.
4. Boost_modeling.py runs the Xgboost and LightGBM. Modeling.py has the code for Logistic regression, SVM, KNN, Random forest and Decision trees algorithms.
5. Figures.ipynb has the scripts written for geenrating the figures displayed in the paper.          
