import pandas as pd
import os
from datetime import datetime

base_path = "/Users/pradnyajadhav/Desktop/psykose"

control_folder = os.path.join(base_path, "control")
patient_folder = os.path.join(base_path, "patient")
control_files = [f for f in os.listdir(control_folder) if f.endswith(".csv")]
patient_files = [f for f in os.listdir(patient_folder) if f.endswith(".csv")]

data = []
def process_files(file_list, class_value):
    for file in file_list:
        df = pd.read_csv(os.path.join(base_path, class_value, file))
        for index, row in df.iterrows():
            timestamp = row['timestamp']
            
            datetime_obj = datetime.strptime(timestamp, "%Y-%m-%d %H:%M:%S")
            
            date = datetime_obj.strftime("%Y-%m-%d")
            time = datetime_obj.strftime("%H:%M")
            activity = row['activity']
            data.append([file, timestamp, date, time, activity, class_value])

process_files(control_files, "control")  
process_files(patient_files, "patient")  

columns = ["Filename", "Timestamp", "Date", "Time", "Activity", "Class"]
master_df = pd.DataFrame(data, columns=columns)
filtered_master_df = master_df

filtered_master_df['Filename'] = filtered_master_df['Filename'].str.replace('.csv', '')

filtered_master_df['Class'] = filtered_master_df['Class'].apply(lambda x: 0 if x == 'control' else 1)
filtered_master_df.to_csv('/Users/pradnyajadhav/Desktop/Schizophrenia_Chronopsychosis/integrated_data.csv', index=False)