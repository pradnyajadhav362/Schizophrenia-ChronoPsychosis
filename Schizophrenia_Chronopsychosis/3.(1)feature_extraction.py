import pandas as pd
import numpy as np
from scipy.stats import kurtosis, skew, entropy
from scipy.signal import find_peaks
from scipy.stats import sem
from statsmodels.tsa.stattools import acf
required_file = '/Users/pradnyajadhav/Desktop/Schizophrenia_Chronopsychosis/required_data.csv'
df_partition = pd.read_csv(required_file)
df_partition['Timestamp'] = pd.to_datetime(df_partition['Timestamp'], format='%Y-%m-%d %H:%M:%S')

time_ranges_3_parts = {
    1: ('00:00:00', '07:59:59'),
    2: ('08:00:00', '15:59:59'),
    3: ('16:00:00', '23:59:59')
}
time_ranges_4_parts = {
    1: ('00:00:00', '05:59:59'),
    2: ('06:00:00', '11:59:59'),
    3: ('12:00:00', '17:59:59'),
    4: ('18:00:00', '23:59:59')
}
time_ranges_6_parts = {
    1: ('00:00:00', '03:59:59'),
    2: ('04:00:00', '07:59:59'),
    3: ('08:00:00', '11:59:59'),
    4: ('12:00:00', '15:59:59'),
    5: ('16:00:00', '19:59:59'),
    6: ('20:00:00', '23:59:59')
}
time_ranges_8_parts = {
    1: ('00:00:00', '02:59:59'),
    2: ('03:00:00', '05:59:59'),
    3: ('06:00:00', '08:59:59'),
    4: ('09:00:00', '11:59:59'),
    5: ('12:00:00', '14:59:59'),
    6: ('15:00:00', '17:59:59'),
    7: ('18:00:00', '20:59:59'),
    8: ('21:00:00', '23:59:59')
}
time_ranges_12_parts = {
    1: ('00:00:00', '01:59:59'),
    2: ('02:00:00', '03:59:59'),
    3: ('04:00:00', '05:59:59'),
    4: ('06:00:00', '07:59:59'),
    5: ('08:00:00', '09:59:59'),
    6: ('10:00:00', '11:59:59'),
    7: ('12:00:00', '13:59:59'),
    8: ('14:00:00', '15:59:59'),
    9: ('16:00:00', '17:59:59'),
    10: ('18:00:00', '19:59:59'),
    11: ('20:00:00', '21:59:59'),
    12: ('22:00:00', '23:59:59')
}
time_ranges_full_day = {
    1: ('00:00:00', '23:59:59')
}
def assign_part_of_day(time, time_ranges):
    for part, (start_time, end_time) in time_ranges.items():
        if start_time <= time.strftime('%H:%M:%S') <= end_time:
            return f'Part {part}'
    return None
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
def process_time_intervals(time_ranges, output_csv):
    df_partition['partofday'] = df_partition['Timestamp'].apply(lambda x: assign_part_of_day(x, time_ranges))
    grouped = df_partition.groupby(['Filename', 'Date', 'partofday'])
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
    df_merged = pd.merge(df_partition, stats, on=['Filename', 'Date', 'partofday'])
    df_merged = df_merged.loc[:, ~df_merged.columns.str.contains('^Unnamed')]
    df_unique = df_merged.drop_duplicates(['Filename', 'Date', 'partofday'])
    result_df = df_unique[['Filename', 'Date', 'Class']].drop_duplicates().reset_index(drop=True)
    part_numbers = list(time_ranges.keys())
    for part in part_numbers:
        part_name = f'part{part}'
        part_df = df_unique[df_unique['partofday'] == f'Part {part}']
        if 'mean' in part_df.columns:
         result_df[f'{part_name}_mean'] = part_df['mean'].values
        
        #result_df[f'{part_name}_mean'] = part_df['mean'].values
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
    column_order = ['Filename', 'Date', 'Class']
    for part in part_numbers:
        part_name = f'part{part}'
        column_order.append(f'{part_name}_mean')
    for part in part_numbers:
        part_name = f'part{part}'
        column_order.append(f'{part_name}_median')
    for part in part_numbers:
        part_name = f'part{part}'
        column_order.append(f'{part_name}_std')
    for part in part_numbers:
        part_name = f'part{part}'
        column_order.append(f'{part_name}_proportion_zeros')
    for part in part_numbers:
        part_name = f'part{part}'
        column_order.append(f'{part_name}_skew')
    for part in part_numbers:
        part_name = f'part{part}'
        column_order.append(f'{part_name}_kurtosis')
    for part in part_numbers:
        part_name = f'part{part}'
        column_order.append(f'{part_name}_max')
    for part in part_numbers:
        part_name = f'part{part}'
        column_order.append(f'{part_name}_mad')
    for part in part_numbers:
        part_name = f'part{part}'
        column_order.append(f'{part_name}_iqr')
    for part in part_numbers:
        part_name = f'part{part}'
        column_order.append(f'{part_name}_cv')
    for part in part_numbers:
        part_name = f'part{part}'
        column_order.append(f'{part_name}_entropy')
    for part in part_numbers:
        part_name = f'part{part}'
        column_order.append(f'{part_name}_autocorr')
    for part in part_numbers:
        part_name = f'part{part}'
        column_order.append(f'{part_name}_peaks')
    for part in part_numbers:
        part_name = f'part{part}'
        column_order.append(f'{part_name}_troughs')
    for part in part_numbers:
        part_name = f'part{part}'
        column_order.append(f'{part_name}_semivariance')
    for part in part_numbers:
        part_name = f'part{part}'
        column_order.append(f'{part_name}_rms')
    result_df = result_df[column_order]
    result_df = result_df.loc[:, ~result_df.columns.str.contains('^Unnamed')]
    result_df.to_csv(output_csv, index=False)

process_time_intervals(time_ranges_3_parts, "3_parts_features.csv")
process_time_intervals(time_ranges_4_parts, "4_parts_features.csv")
process_time_intervals(time_ranges_6_parts, "6_parts_features.csv")
process_time_intervals(time_ranges_8_parts, "8_parts_features.csv")
process_time_intervals(time_ranges_12_parts, "12_parts_features.csv")
process_time_intervals(time_ranges_full_day, "full_day_features.csv")