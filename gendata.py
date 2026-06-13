# -*- coding: utf-8 -*-
"""
Created on Thu Jun 11 20:32:45 2026

@author: acer
"""

import numpy as np
import csv

# Parameters
f = 12            # frequency in Hz
Fs = 1000         # sampling frequency in Hz
T = 1             # duration in seconds
t = np.linspace(0, T, int(Fs*T), endpoint=False)

# Generate sinusoidal signal
signal = np.sin(2 * np.pi * f * t)

# Save to CSV
with open("sinusoidal_12Hz.csv", "w", newline="") as file:
    writer = csv.writer(file)
    writer.writerow(["Time (s)", "Signal"])
    for ti, si in zip(t, signal):
        writer.writerow([ti, si])

print("CSV file 'sinusoidal_12Hz.csv' generated successfully.")
