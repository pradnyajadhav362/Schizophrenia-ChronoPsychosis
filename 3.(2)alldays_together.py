import pandas as pd
import numpy as np
from scipy.stats import kurtosis, skew, entropy
from scipy.signal import find_peaks
from scipy.stats import sem
from statsmodels.tsa.stattools import acf

required_file = '/Users/pradnyajadhav/Desktop/Schizophrenia_Chronopsychosis/required_data.csv'
df = pd.read_csv(required_file)

# Convert 'Timestamp' to datetime
df['Timestamp'] = pd.to_datetime(df['Timestamp'], format='%Y-%m-%d %H:%M:%S')

# Group by 'Filename' and calculate features for each unique filename
grouped = df.groupby('Filename')

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

def calculate_cv(data):
    return np.std(data) / np.mean(data)

# Calculate features for each unique filename
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
    'Filename', 'mean', 'median', 'std', 'mad',
    'proportion_zeros', 'skew', 'kurtosis', 'max', 'entropy',
    'autocorr', 'peaks', 'troughs', 'semivariance', 'rms', 'iqr', 'cv'
]

# Add 'Date' and 'Class' columns
stats[['Date', 'Class']] = grouped[['Date', 'Class']].first().reset_index()[['Date', 'Class']]
# Save the calculated features to a CSV file
stats.to_csv("/Users/pradnyajadhav/Desktop/Schizophrenia_Chronopsychosis/all_filenames_features.csv", index=False)
