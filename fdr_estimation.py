import numpy as np
import pandas as pd
import scipy.io
from scipy.stats import norm, binom
import matplotlib.pyplot as plt

# Load the true peptide spectra from the decoy.mat file
decoy_data = scipy.io.loadmat('decoy.mat')
trueset = decoy_data['trueset']

# Read in the TSV file containing the spectra and m/z values
data = pd.read_csv('syntetic_exp_SN1p0_1-27_centroid (2)_ms1.tsv', sep='\t')

# Set the ppm tolerance value
ppmtol = 10

# Identify true positives, false positives, and decoy spectra
tpindex = []
fpindex = []
decoyindex = []
for i in range(data.shape[0]):
    mz = data.iloc[i]['m/z']
    is_decoy = data.iloc[i]['decoy']
    if is_decoy:
        decoyindex.append(i)
    else:
        # Compare the m/z value to the true values in the trueset file
        diff = np.abs(trueset - mz)
        idx = np.argmin(diff)
        min_diff = diff[idx]
        ppm_diff = min_diff / trueset[idx] * 1e6
        if ppm_diff <= ppmtol:
            tpindex.append(i)
        else:
            fpindex.append(i)

# Calculate QScores for true positives, false positives, and decoy spectra
q_tp = data.iloc[tpindex]['QScore']
q_fp = data.iloc[fpindex]['QScore']
q_decoy = data.iloc[decoyindex]['QScore']

# Plot histograms of QScores for false positives and decoy spectra
plt.hist(q_fp, bins=50, alpha=0.5, label='False Positives')
plt.hist(q_decoy, bins=50, alpha=0.5, label='Decoys')
plt.legend()
plt.xlabel('QScore')
plt.ylabel('Count')
plt.show()

# Calculate q-values for each QScore using the CDF of the QScore for true spectra and decoy spectra
dpv = np.histogram(q_tp, bins=50, range=(0, 1), density=True)[0]
fpv = np.histogram(q_fp, bins=50, range=(0, 1), density=True)[0]
cdpv = np.cumsum(dpv)
cfpv = np.cumsum(fpv)
ks_stat, p_value = scipy.stats.ks_2samp(q_tp, q_fp)

# Plot estimated FDR and q-value as a function of QScore
fdr = cfpv / cdpv
qvalue = np.minimum.accumulate(fdr[::-1])[::-1]
plt.plot(qvalue, label='q-value')
plt.plot(fdr, label='FDR')
plt.legend()
plt.xlabel('QScore')
plt.ylabel('Estimated FDR and q-value')
plt.show()
