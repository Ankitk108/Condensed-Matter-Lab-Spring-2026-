# Curie Temperature Experiment

This experiment determines the Curie temperature (Tc) of ferromagnetic samples
using the four-probe method.

## Objective
To determine Tc by analyzing resistance variation with temperature.

## Method
1. Measure Vdc across sample.
2. Maintain constant current.
3. Plot R (or Vdc) vs T.
4. Compute dR/dT vs T.
5. Tc = Temperature at maximum dR/dT.

## Run
pip install numpy pandas matplotlib scipy
python analysis.py
