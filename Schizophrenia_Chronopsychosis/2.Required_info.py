import pandas as pd
df = pd.read_csv('/Users/pradnyajadhav/Desktop/Schizophrenia_Chronopsychosis/integrated_data.csv')


df['datetime'] = pd.to_datetime(df['Date'] + ' ' + df['Time'])

selected_rows = []

for filename in df['Filename'].unique():
    
    start_condition = (df['Filename'] == filename) & (df['datetime'].dt.time == pd.to_datetime('00:00').time())
    end_condition = (df['Filename'] == filename) & (df['datetime'].dt.time == pd.to_datetime('23:59').time())
    dates_with_start = df.loc[start_condition, 'Date'].unique()
    dates_with_end = df.loc[end_condition, 'Date'].unique()
    
    dates_with_both_times = sorted(set(dates_with_start) & set(dates_with_end))
    selected_rows.append(df[(df['Filename'] == filename) & (df['Date'].isin(dates_with_both_times))])
    

fulldays_df = pd.concat(selected_rows)

num_of_dates = {
    'patient_1': 13,
    'patient_10': 14,
    'patient_11': 13,
    'patient_12': 13,
    'patient_13': 12,
    'patient_14': 13,
    'patient_15': 13,
    'patient_16': 13,
    'patient_17': 13,
    'patient_18': 13,
    'patient_19': 13,
    'patient_2': 13,
    'patient_20': 13,
    'patient_21': 13,
    'patient_22': 13,
    'patient_3': 13,
    'patient_4': 13,
    'patient_5': 13,
    'patient_6': 13,
    'patient_7': 12,
    'patient_8': 13,
    'patient_9': 13,
    'control_1': 8,
    'control_10': 8,
    'control_11': 13,
    'control_12': 14,
    'control_13': 13,
    'control_14': 13,
    'control_15': 11,
    'control_16': 13,
    'control_17': 9,
    'control_18': 13,
    'control_19': 13,
    'control_2': 20,
    'control_20': 13,
    'control_21': 8,
    'control_22': 13,
    'control_23': 13,
    'control_24': 13,
    'control_25': 13,
    'control_26': 13,
    'control_27': 13,
    'control_28': 16,
    'control_29': 13,
    'control_3': 12,
    'control_30': 9,
    'control_31': 13,
    'control_32': 14,
    'control_4': 13,
    'control_5': 13,
    'control_6': 13,
    'control_7': 13,
    'control_8': 13,
    'control_9': 13,
}

required_rows = []

for filename, num_dates in num_of_dates.items():
    filename_condition = (fulldays_df['Filename'] == filename)
    unique_dates = fulldays_df.loc[filename_condition, 'Date'].unique()[:num_dates]
    required_rows.append(fulldays_df[filename_condition & df['Date'].isin(unique_dates)])

required_df = pd.concat(required_rows)
print(required_df)
required_df.to_csv('required_data.csv', index=False)
