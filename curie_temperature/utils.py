import pandas as pd
import numpy as np

def load_data(filepath):
    df = pd.read_csv(filepath)

    # Remove obvious typo outliers (e.g., 3.9 V mistake)
    df = df[df["Vdc_V"] < 3.5]

    # Average duplicate temperatures
    df = df.groupby("Temperature_C", as_index=False).mean()

    return df


def apply_vtotal_method(df):
    """
    Apply required method:
    1. Compute V_total_avg from rows where both Vdc and Vsc exist
    2. For missing Vsc:
       Vsc = V_total_avg - Vdc
    """

    valid = df.dropna(subset=["Vdc_V", "Vsc_V"]).copy()
    valid["V_total"] = valid["Vdc_V"] + valid["Vsc_V"]

    V_total_avg = valid["V_total"].mean()

    df["Vsc_V"] = df.apply(
        lambda row: V_total_avg - row["Vdc_V"]
        if pd.isna(row["Vsc_V"]) else row["Vsc_V"],
        axis=1
    )

    return df, V_total_avg


def compute_capacitance(df):
    df["C"] = df["Vsc_V"] / df["Vdc_V"]

    # Reference capacitance below 60°C
    C_ref = df[df["Temperature_C"] < 60]["C"].mean()

    df["epsilon_r"] = df["C"] / C_ref

    return df, C_ref
