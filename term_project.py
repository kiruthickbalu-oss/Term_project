import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import warnings
from scipy.optimize import differential_evolution
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

warnings.filterwarnings("ignore")

# --- CONFIGURATION ---
# Place your NASA csv file in a folder named 'data'
DATA_PATH = os.path.join('data', 'battery_data.csv') 
OUTPUT_DIR = 'results'

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

# --- CORE FUNCTIONS ---

def load_and_clean_data(filepath):
    """Loads raw NASA data and prepares it for modeling."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Data file not found at {filepath}. Please add it to the data/ folder.")
    
    df = pd.read_csv(filepath)
    # Ensure numeric time and clean missing values
    df['time'] = pd.to_numeric(df['time'], errors='coerce')
    df = df.dropna(subset=['time', 'voltage_load', 'current_load']).reset_index(drop=True)
    
    # Isolate the first discharge cycle for analysis
    time_sec_all = df['time'].values
    dt_all = np.diff(time_sec_all, prepend=time_sec_all[0])
    dt_all[0] = 1.0
    
    # Identify cycle breaks (long time gaps)
    cycle_splits = np.where(dt_all > 100)[0]
    end_idx = cycle_splits[0] if len(cycle_splits) > 0 else len(df)
    
    df_cycle = df.iloc[:end_idx].copy()
    
    # Feature extraction
    time_sec = df_cycle['time'].values
    dt = np.diff(time_sec, prepend=time_sec[0])
    dt[0] = 1.0
    I_meas = df_cycle['current_load'].values
    V_meas = df_cycle['voltage_load'].values
    
    # Calculate Capacity (Amp-seconds) and True SOC
    Q_nom_As = np.sum(I_meas * dt)
    soc_true = 1.0 - (np.cumsum(I_meas * dt) / Q_nom_As)
    soc_true = np.clip(soc_true, 0.0, 1.0)
    
    return time_sec, dt, I_meas, V_meas, soc_true, Q_nom_As

def poly_ocv(soc, p):
    """5th-order Polynomial Open Circuit Voltage model."""
    return p[0] + p[1]*soc + p[2]*(soc**2) + p[3]*(soc**3) + p[4]*(soc**4) + p[5]*(soc**5)

def get_ocv_derivative(soc, p):
    """Derivative of OCV for the Kalman Filter H matrix."""
    return p[1] + 2*p[2]*soc + 3*p[3]*(soc**2) + 4*p[4]*(soc**3) + 5*p[5]*(soc**4)

def fit_ocv_curve(soc_true, V_meas, I_meas):
    """Estimates OCV parameters using Differential Evolution."""
    def cost(p):
        # Assumes a baseline R0 of 0.05 for initial fitting
        v_sim = poly_ocv(soc_true, p) - I_meas * 0.05
        return np.mean((v_sim - V_meas)**2)
    
    bounds = [(3.0, 4.5), (-2, 2), (-5, 5), (-5, 5), (-5, 5), (-5, 5)]
    res = differential_evolution(cost, bounds, seed=42)
    return res.x

# --- MODEL CLASSES ---

class AEKF_SOC_Estimator:
    """Adaptive Extended Kalman Filter for SOC Estimation."""
    def __init__(self, params, p_ocv, Q_nom_As):
        self.R0, self.R1, self.C1, self.R2, self.C2 = params
        self.p_ocv = p_ocv
        self.Q_nom = Q_nom_As
        self.x = np.array([[1.0], [0.0], [0.0]]) # [SOC, U1, U2]
        self.P = np.eye(3) * 1e-2
        self.Q = np.eye(3) * 1e-5
        self.R = np.array([[1e-2]])
        self.window_size = 20
        self.innovations = []

    def step(self, I, V_meas, dt):
        # State Transition Matrix
        A = np.diag([1.0, np.exp(-dt/(self.R1*self.C1)), np.exp(-dt/(self.R2*self.C2))])
        B = np.array([[-dt / self.Q_nom], 
                      [self.R1 * (1 - np.exp(-dt/(self.R1*self.C1)))], 
                      [self.R2 * (1 - np.exp(-dt/(self.R2*self.C2)))]])
        
        # Prediction
        x_pred = A @ self.x + B * I
        P_pred = A @ self.P @ A.T + self.Q
        
        # Measurement Update
        soc_pred = x_pred[0, 0]
        ocv_pred = poly_ocv(soc_pred, self.p_ocv)
        V_pred = ocv_pred - I * self.R0 - x_pred[1, 0] - x_pred[2, 0]
        
        dOCV = get_ocv_derivative(soc_pred, self.p_ocv)
        H = np.array([[dOCV, -1.0, -1.0]])
        e = V_meas - V_pred

        # Adaptive Noise R estimation
        self.innovations.append(e)
        if len(self.innovations) > self.window_size: self.innovations.pop(0)
        if len(self.innovations) == self.window_size:
            self.R[0,0] = max(np.var(self.innovations) - (H @ P_pred @ H.T)[0,0], 1e-4)

        S = H @ P_pred @ H.T + self.R
        K = P_pred @ H.T @ np.linalg.inv(S)
        self.x = x_pred + K * e
        self.P = (np.eye(3) - K @ H) @ P_pred
        self.x[0,0] = np.clip(self.x[0,0], 0.0, 1.0)
        return self.x[0,0]

# --- MAIN EXECUTION ---

if __name__ == "__main__":
    print("Starting Battery SOC Estimation Pipeline...")
    
    # 1. Data Prep
    time_sec, dt, I_meas, V_meas, soc_true, Q_nom_As = load_and_clean_data(DATA_PATH)
    
    # 2. Model Identification
    print("Fitting OCV and identifying parameters...")
    p_ocv = fit_ocv_curve(soc_true, V_meas, I_meas)
    # Placeholder for global params (In real use, these come from your GA identification function)
    ecm_params = [0.05, 0.01, 2000, 0.02, 4000] 
    
    # 3. SOC Estimation
    print("Running AEKF...")
    aekf = AEKF_SOC_Estimator(ecm_params, p_ocv, Q_nom_As)
    soc_estimated = np.array([aekf.step(I_meas[k], V_meas[k], dt[k]) for k in range(len(I_meas))])

    # 4. Visualization
    plt.figure(figsize=(12, 6))
    plt.plot(time_sec, soc_true * 100, 'k', label='True SOC (Coulomb Counting)')
    plt.plot(time_sec, soc_estimated * 100, 'r--', label='Estimated SOC (AEKF)')
    plt.title("SOC Tracking Performance")
    plt.xlabel("Time (s)")
    plt.ylabel("SOC (%)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    save_path = os.path.join(OUTPUT_DIR, 'soc_estimation_results.png')
    plt.savefig(save_path)
    print(f"Success! Plot saved to {save_path}")
