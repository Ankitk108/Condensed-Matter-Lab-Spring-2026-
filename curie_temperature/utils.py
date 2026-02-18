import pandas as pd
import numpy as np
from scipy.signal import savgol_filter
from config import SMOOTH_WINDOW, SMOOTH_POLYORDER

def load_data(filepath):
    df = pd.read_csv(filepath)
    df = df.dropna(subset=["Vdc_V"])
    df = df[df["Vdc_V"] < 3.5]
    df = df.sort_values("Temperature_C")
    return df

def compute_derivative(T, V):
    V_smooth = savgol_filter(V, SMOOTH_WINDOW, SMOOTH_POLYORDER)
    dVdT = np.gradient(V_smooth, T)
    return V_smooth, dVdT

def find_tc(T, derivative):
    idx = np.argmax(derivative)
    return T[idx]
