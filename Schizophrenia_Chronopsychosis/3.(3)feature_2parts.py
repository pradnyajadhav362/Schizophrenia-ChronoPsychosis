import pandas as pd
import numpy as np
from scipy.stats import kurtosis
from scipy.stats import kurtosis, skew, entropy
from scipy.signal import find_peaks
from scipy.stats import sem
from statsmodels.tsa.stattools import acf

input_file_path = '/Users/pradnyajadhav/Desktop/Schizophrenia_Chronopsychosis/required_data.csv'
df_allbasic = pd.read_csv(input_file_path)

df_allbasic['Timestamp'] = pd.to_datetime(df_allbasic['Timestamp'], format='%Y-%m-%d %H:%M:%S')

time_ranges = {
    1: ('08:00:00', '19:59:59')
}

def assign_part_of_day(time):
    time_str = time.strftime('%H:%M:%S')
    for part, (start_time, end_time) in time_ranges.items():
        if start_time <= time_str <= end_time:
            return f'Part {part}'
    return 'Part 2'

df_allbasic['partofday'] = df_allbasic['Timestamp'].apply(assign_part_of_day)
grouped = df_allbasic.groupby(['Filename', 'Date', 'partofday'])

def calculate_mad(data):
    median = np.median(data)
    mad = np.median(np.abs(data - median))
    return mad
def calculate_entropy(data):
    return entropy(data)
def calculate_autocorr(data):
    return acf(data, fft=False)[1]
def calculate_peaks(data):
    peaks, _ = find_peaks(data)
    return len(peaks)
def calculate_troughs(data):
    troughs, _ = find_peaks(-data)
    return len(troughs)
def calculate_semivariance(data):
    mean = np.mean(data)
    return np.mean((data - mean) ** 2)
def calculate_rms(data):
    return np.sqrt(np.mean(data ** 2))

stats = grouped['Activity'].agg(
        mean='mean',
        median='median',
        std='std',
        mad=calculate_mad,
        proportion_zeros=lambda x: (x == 0).mean(),
        skew='skew',
        kurtosis=lambda x: kurtosis(x),
        max='max',
        entropy=calculate_entropy,
        autocorr=calculate_autocorr,
        peaks=calculate_peaks,
        troughs=calculate_troughs,
        semivariance=calculate_semivariance,
        rms=calculate_rms,
        iqr=lambda x: np.percentile(x, 75) - np.percentile(x, 25),
    ).reset_index()
stats['cv'] = stats['std'] / stats['mean']
stats.columns = [
        'Filename', 'Date', 'partofday', 'mean', 'median', 'std', 'mad',
        'proportion_zeros', 'skew', 'kurtosis', 'max', 'entropy',
        'autocorr', 'peaks', 'troughs', 'semivariance', 'rms', 'iqr', 'cv'
    ]

df_merged = pd.merge(df_allbasic, stats, on=['Filename', 'Date', 'partofday'])
df_merged = df_merged.loc[:, ~df_merged.columns.str.contains('^Unnamed')]

df_unique = df_merged.drop_duplicates(['Filename', 'Date', 'partofday'])

part_numbers = [1, 2]
result_df = df_unique[['Filename', 'Date', 'Class']].drop_duplicates().reset_index(drop=True)

for part in part_numbers:
    part_name = f'part{part}'
    part_df = df_unique[df_unique['partofday'] == f'Part {part}']
    result_df[f'{part_name}_mean'] = part_df['mean'].values
    result_df[f'{part_name}_median'] = part_df['median'].values
    result_df[f'{part_name}_std'] = part_df['std'].values
    result_df[f'{part_name}_proportion_zeros'] = part_df['proportion_zeros'].values
    result_df[f'{part_name}_skew'] = part_df['skew'].values
    result_df[f'{part_name}_kurtosis'] = part_df['kurtosis'].values
    result_df[f'{part_name}_max'] = part_df['max'].values
    result_df[f'{part_name}_mad'] = part_df['mad'].values
    result_df[f'{part_name}_iqr'] = part_df['iqr'].values
    result_df[f'{part_name}_cv'] = part_df['cv'].values
    result_df[f'{part_name}_entropy'] = part_df['entropy'].values
    result_df[f'{part_name}_autocorr'] = part_df['autocorr'].values
    result_df[f'{part_name}_peaks'] = part_df['peaks'].values
    result_df[f'{part_name}_troughs'] = part_df['troughs'].values
    result_df[f'{part_name}_semivariance'] = part_df['semivariance'].values
    result_df[f'{part_name}_rms'] = part_df['rms'].values

result_df.to_csv("/Users/pradnyajadhav/Desktop/Schizophrenia_Chronopsychosis/2_parts_features(1).csv", index=False)
