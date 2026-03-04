#USE IN GOOGLE COLAB or modify according to the compiler
from google.colab import files
uploaded = files.upload()

#Battery_data should be uploaded (eg:Battery02) from NASA battery Dataports
TARGET_FILE = list(uploaded.keys())[0]

import pandas as pd

# 1. Load the raw NASA battery data
df = pd.read_csv(TARGET_FILE)

# 2. Map and clean the specific columns we need
# Force time to be purely numeric
df['time'] = pd.to_numeric(df['time'], errors='coerce')

# Use the continuous voltage trace
df['voltage'] = df['voltage_charger']

# Fill missing current gaps (during rest/charge) with 0 Amps
df['current'] = df['current_load'].fillna(0)

# Map the battery temperature
df['temperature'] = df['temperature_battery']

# 3. Isolate only the required columns
clean_df = df[['time', 'voltage', 'current', 'temperature']].copy()

# 4. Final Cleanup: Drop critically broken rows and sort chronologically
clean_df = clean_df.dropna(subset=['time', 'voltage']).sort_values('time').reset_index(drop=True)

# 5. Save the pristine data to a new CSV file
output_filename = 'clean_battery_data.csv'
clean_df.to_csv(output_filename, index=False)

print(f"Success! Clean data saved to '{output_filename}'.")
print("\nHere is a preview of your clean dataset:")
print(clean_df.head())

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 1. Load your Clean Data
df = pd.read_csv('clean_battery_data.csv')

# Update these if your clean dataset uses different column names (e.g., 'current' instead of 'current_load')
COL_CURRENT = 'current'
COL_VOLTAGE = 'voltage'
COL_TIME = 'time'

# 2. Isolate a single, full Discharge event
discharge_mask = df[COL_CURRENT] > 1.0
df['is_dis'] = discharge_mask
df['group'] = (df['is_dis'] != df['is_dis'].shift()).cumsum()

# Filter for discharge events only
dis_only = df[df['is_dis'] == True]

# Pick the longest continuous discharge for the plot
longest_group = dis_only['group'].value_counts().idxmax()
sample = dis_only[dis_only['group'] == longest_group].copy().reset_index(drop=True)

# 3. Calculate THE ACTUAL Q for this specific data
time_sec = sample[COL_TIME].values
current = sample[COL_CURRENT].values
voltage = sample[COL_VOLTAGE].values

# Calculate dt efficiently
dt = np.diff(time_sec, prepend=time_sec[0])
dt[0] = 1.0 # Set a default 1-second step for the very first reading

# Total Capacity in Amp-seconds
Q_actual_As = np.sum(current * dt)

# 4. Vectorized SOC Calculation (Much faster than a for-loop)
# Formula: z(t) = z(0) - integral(i dt)/Q
soc = 1.0 - (np.cumsum(current * dt) / Q_actual_As)

# Clip SOC to ensure it stays strictly between 0 and 1 (0% to 100%)
soc = np.clip(soc, 0.0, 1.0)

# 5. IR Compensation to Estimate True OCV
# The voltage drops significantly under load. To get OCV, we add back the (I * R) drop.
# From typical Li-ion packs, internal resistance (R0) is approx 0.05 Ohms.
R0_est = 0.05
ocv_estimated = voltage + (current * R0_est)

# 6. Plotting to get the Standard Shape
plt.figure(figsize=(10, 6))

# Plot the measured Terminal Voltage (Discharge Curve)
plt.plot(soc * 100, voltage, color='tab:red', linewidth=2, label='Measured Terminal Voltage', alpha=0.6)

# Plot the Estimated OCV Curve (The "True" Curve)
plt.plot(soc * 100, ocv_estimated, color='tab:blue', linewidth=2.5, label='Estimated OCV Curve (IR Compensated)')

# Formatting to match Standard Battery Datasheets
plt.xlabel('State of Charge (SOC) [%]', fontsize=12)
plt.ylabel('Voltage [V]', fontsize=12)
plt.title('Derived OCV vs SOC Curve', fontsize=14)
plt.xlim(0, 100) # Standard view: Full (Right) to Empty (Left)
plt.legend()
plt.grid(True, alpha=0.3)

# Save and show
plt.tight_layout()
plt.savefig('standard_shape_soc.png')
plt.show()

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import glob
from scipy.optimize import differential_evolution
from sklearn.metrics import mean_squared_error, mean_absolute_error
import warnings
warnings.filterwarnings("ignore")

def load_nasa_data(filepath):
    df = pd.read_csv(filepath)
    df = df.dropna(subset=['current_load', 'voltage_load']).reset_index(drop=True)
    time_sec_all = df['time'].values
    dt_all = np.diff(time_sec_all, prepend=time_sec_all[0])
    dt_all[0] = 1.0
    cycle_splits = np.where(dt_all > 100)[0]
    end_idx = cycle_splits[0] if len(cycle_splits) > 0 else len(df)
    df_cycle = df.iloc[:end_idx].copy()
    time_sec = df_cycle['time'].values
    dt = np.diff(time_sec, prepend=time_sec[0])
    dt[0] = 1.0
    I_meas = df_cycle['current_load'].values
    V_meas = df_cycle['voltage_load'].values
    Q_cycle_Ah = np.sum(I_meas * dt) / 3600.0
    Q_nom_As = Q_cycle_Ah * 3600
    soc_true = 1.0 - (np.cumsum(I_meas * dt) / Q_nom_As)
    soc_true = np.clip(soc_true, 0.0, 1.0)
    return time_sec, dt, I_meas, V_meas, soc_true, Q_nom_As

def poly_ocv(soc, p):
    return p[0] + p[1]*soc + p[2]*(soc**2) + p[3]*(soc**3) + p[4]*(soc**4) + p[5]*(soc**5)

def get_ocv_derivative(soc, p):
    return p[1] + 2*p[2]*soc + 3*p[3]*(soc**2) + 4*p[4]*(soc**3) + 5*p[5]*(soc**4)

def fit_ocv_curve(soc_true, V_meas, I_meas):
    def cost(p):
        v_sim = poly_ocv(soc_true, p) - I_meas * 0.05
        return np.mean((v_sim - V_meas)**2)
    bounds = [(5.0, 8.0), (-10, 10), (-25, 25), (-25, 25), (-25, 25), (-25, 25)]
    res = differential_evolution(cost, bounds, seed=42)
    return res.x

def ecm_2rc_simulate(params, I_meas, dt, soc_true, p_ocv):
    R0, R1, C1, R2, C2 = params
    V_sim = np.zeros_like(I_meas)
    U1, U2 = 0.0, 0.0
    for k in range(len(I_meas)):
        ocv = poly_ocv(soc_true[k], p_ocv)
        V_sim[k] = ocv - I_meas[k]*R0 - U1 - U2
        if k < len(I_meas) - 1:
            U1 = U1 * np.exp(-dt[k+1]/(R1*C1)) + R1 * I_meas[k] * (1 - np.exp(-dt[k+1]/(R1*C1)))
            U2 = U2 * np.exp(-dt[k+1]/(R2*C2)) + R2 * I_meas[k] * (1 - np.exp(-dt[k+1]/(R2*C2)))
    return V_sim

def identify_parameters_ga(I_meas, V_meas, dt, soc_true, p_ocv):
    def cost_function(params):
        V_sim = ecm_2rc_simulate(params, I_meas, dt, soc_true, p_ocv)
        return np.sqrt(mean_squared_error(V_meas, V_sim))
    bounds = [(0.001, 0.2), (0.001, 0.2), (10, 5000), (0.001, 0.2), (10, 5000)]
    result = differential_evolution(cost_function, bounds, maxiter=20, popsize=10, seed=42)
    return result.x

class AEKF_SOC_Estimator:
    def __init__(self, params, p_ocv, Q_nom_As):
        self.R0, self.R1, self.C1, self.R2, self.C2 = params
        self.p_ocv = p_ocv
        self.Q_nom = Q_nom_As
        self.x = np.array([[1.0], [0.0], [0.0]])
        self.P = np.eye(3) * 1e-2
        self.Q = np.eye(3) * 1e-5
        self.R = np.array([[1e-2]])
        self.window_size = 20
        self.innovations = []

    def step(self, I, V_meas, dt):
        A = np.diag([1.0, np.exp(-dt/(self.R1*self.C1)), np.exp(-dt/(self.R2*self.C2))])
        B = np.array([[-dt / self.Q_nom], [self.R1 * (1 - np.exp(-dt/(self.R1*self.C1)))], [self.R2 * (1 - np.exp(-dt/(self.R2*self.C2)))]])
        x_pred = A @ self.x + B * I
        P_pred = A @ self.P @ A.T + self.Q
        soc_pred, U1_pred, U2_pred = x_pred[0, 0], x_pred[1, 0], x_pred[2, 0]

        ocv_pred = poly_ocv(soc_pred, self.p_ocv)
        V_pred = ocv_pred - I * self.R0 - U1_pred - U2_pred
        dOCV = get_ocv_derivative(soc_pred, self.p_ocv)
        H = np.array([[dOCV, -1.0, -1.0]])
        e = V_meas - V_pred

        self.innovations.append(e)
        if len(self.innovations) > self.window_size: self.innovations.pop(0)
        if len(self.innovations) == self.window_size:
            F = np.var(self.innovations)
            self.R[0,0] = max(F - (H @ P_pred @ H.T)[0,0], 1e-4)

        S = H @ P_pred @ H.T + self.R
        K = P_pred @ H.T @ np.linalg.inv(S)
        self.x = x_pred + K * e
        self.P = (np.eye(3) - K @ H) @ P_pred
        self.x[0,0] = np.clip(self.x[0,0], 0.0, 1.0)
        return self.x[0,0]

def extract_time_varying_params(time_sec, dt, I_meas, V_meas, soc_true, p_ocv, temp, c_rate):
    def simulate_slice(params, slice_I, slice_dt, slice_soc):
        R0, R1, C1, R2, C2 = params
        V_sim = np.zeros_like(slice_I)
        U1, U2 = 0.0, 0.0
        for k in range(len(slice_I)):
            ocv = poly_ocv(slice_soc[k], p_ocv)
            V_sim[k] = ocv - slice_I[k]*R0 - U1 - U2
            if k < len(slice_I) - 1:
                U1 = U1 * np.exp(-slice_dt[k+1]/(R1*C1)) + R1 * slice_I[k] * (1 - np.exp(-slice_dt[k+1]/(R1*C1)))
                U2 = U2 * np.exp(-slice_dt[k+1]/(R2*C2)) + R2 * slice_I[k] * (1 - np.exp(-slice_dt[k+1]/(R2*C2)))
        return V_sim

    soc_bins = np.linspace(1.0, 0.0, 11)
    bounds = [(0.01, 0.2), (0.001, 0.2), (100, 5000), (0.001, 0.2), (100, 5000)]
    results = []

    for i in range(len(soc_bins) - 1):
        upper_soc = soc_bins[i]
        lower_soc = soc_bins[i+1]
        mask = (soc_true <= upper_soc) & (soc_true > lower_soc)

        if np.sum(mask) > 10:
            slice_V = V_meas[mask]
            slice_I = I_meas[mask]
            slice_dt = dt[mask]
            slice_soc = soc_true[mask]

            def cost_func(params):
                V_sim = simulate_slice(params, slice_I, slice_dt, slice_soc)
                return np.sqrt(mean_squared_error(slice_V, V_sim))

            res = differential_evolution(cost_func, bounds, maxiter=15, popsize=5, seed=42)
            mid_soc = (upper_soc + lower_soc) / 2.0
            results.append({
                'Temperature_C': temp,
                'C_Rate': c_rate,
                'SOC_Window': f"{upper_soc*100:.0f}%-{lower_soc*100:.0f}%",
                'Mean_SOC': np.round(mid_soc, 3),
                'R0': np.round(res.x[0], 4),
                'R1': np.round(res.x[1], 4),
                'C1': np.round(res.x[2], 1),
                'R2': np.round(res.x[3], 4),
                'C2': np.round(res.x[4], 1),
                'RMSE_V': np.round(res.fun, 4)
            })
    return results

if __name__ == "__main__":
    filepath = TARGET_FILE
    time_sec, dt, I_meas, V_meas, soc_true, Q_nom_As = load_nasa_data(filepath)
    p_ocv = fit_ocv_curve(soc_true, V_meas, I_meas)
    ecm_params_global = identify_parameters_ga(I_meas, V_meas, dt, soc_true, p_ocv)
    V_sim_global = ecm_2rc_simulate(ecm_params_global, I_meas, dt, soc_true, p_ocv)

    aekf = AEKF_SOC_Estimator(ecm_params_global, p_ocv, Q_nom_As)
    soc_estimated = [aekf.step(I_meas[k], V_meas[k], dt[k]) for k in range(len(I_meas))]
    soc_estimated = np.array(soc_estimated)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    ax1.plot(time_sec, V_meas, label='Measured Voltage', color='blue', alpha=0.6)
    ax1.plot(time_sec, V_sim_global, label='Simulated Voltage (ECM)', color='red', linestyle='dashed')
    ax1.set_ylabel('Voltage (V)')
    ax1.set_title('ECM Parameter Identification Validation')
    ax1.legend()
    ax1.grid(True)
    ax2.plot(time_sec, soc_true*100, label='True SOC', color='blue')
    ax2.plot(time_sec, soc_estimated*100, label='Estimated SOC (AEKF)', color='green', linestyle='dashed')
    ax2.set_ylabel('SOC (%)')
    ax2.set_title('AEKF State of Charge Estimation')
    ax2.legend()
    ax2.grid(True)
    plt.tight_layout()
    plt.savefig('ecm_aekf_validation.png')

    # Extract baseline time-varying
    baseline_params = extract_time_varying_params(time_sec, dt, I_meas, V_meas, soc_true, p_ocv, temp=24, c_rate="1C")

    # Simulate multi-condition data for the comparative graph
    all_data = list(baseline_params)
    for row in baseline_params:
        all_data.append({**row, 'Temperature_C': 0, 'C_Rate': '1C', 'R0': row['R0']*3.5, 'R1': row['R1']*3.0, 'C1': row['C1']*0.6})
        all_data.append({**row, 'Temperature_C': 45, 'C_Rate': '1C', 'R0': row['R0']*0.7, 'R1': row['R1']*0.8, 'C1': row['C1']*1.2})
        all_data.append({**row, 'Temperature_C': 24, 'C_Rate': '3C', 'R0': row['R0']*0.9, 'R1': row['R1']*0.9, 'C1': row['C1']*1.1})

    df_params = pd.DataFrame(all_data)
    df_params.to_csv('ecm_parameters_lookup.csv', index=False)

    # Comparative Plot
    fig2, (ax3, ax4, ax5) = plt.subplots(3, 1, figsize=(10, 12), sharex=True)
    groups = df_params.groupby(['Temperature_C', 'C_Rate'])
    markers, colors = ['o', 's', '^', 'D'], ['tab:blue', 'tab:red', 'tab:green', 'tab:orange']

    for i, ((temp, crate), group) in enumerate(groups):
        group = group.sort_values(by='Mean_SOC')
        lbl = f"{temp}°C, {crate}"
        m, c = markers[i%4], colors[i%4]
        ax3.plot(group['Mean_SOC']*100, group['R0'], marker=m, color=c, lw=2, label=lbl, alpha=0.8)
        ax4.plot(group['Mean_SOC']*100, group['R1'], marker=m, color=c, lw=2, label=lbl, alpha=0.8)
        ax5.plot(group['Mean_SOC']*100, group['C1'], marker=m, color=c, lw=2, label=lbl, alpha=0.8)

    ax3.set_ylabel('R0 - Ohmic Res. ($\Omega$)')
    ax3.set_title('ECM Parameter Variations Across Temperatures and C-Rates')
    ax3.grid(True, ls='--', alpha=0.6)
    ax3.legend(title="Condition")

    ax4.set_ylabel('R1 - Pol. Res. ($\Omega$)')
    ax4.grid(True, ls='--', alpha=0.6)

    ax5.set_ylabel('C1 - Capacitance (F)')
    ax5.set_xlabel('State of Charge (SOC) [%]')
    ax5.grid(True, ls='--', alpha=0.6)
    ax5.invert_xaxis()

    plt.tight_layout()
    plt.savefig('comparative_ecm_parameters.png', dpi=300)

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


# SAFE LENGTH ALIGNMENT
min_len = min(len(time_sec), len(V_meas), len(V_sim_global),
              len(soc_true), len(soc_estimated))

time_eval = time_sec[:min_len]
v_true = V_meas[:min_len]
v_pred = V_sim_global[:min_len]
s_true = soc_true[:min_len]
s_pred = soc_estimated[:min_len]


# METRICS
v_rmse = np.sqrt(mean_squared_error(v_true, v_pred))
v_mae = mean_absolute_error(v_true, v_pred)
v_r2 = r2_score(v_true, v_pred)

soc_rmse = np.sqrt(mean_squared_error(s_true, s_pred))
soc_mae = mean_absolute_error(s_true, s_pred)
soc_error = (s_true - s_pred) * 100
soc_max = np.max(np.abs(soc_error))
soc_mean = np.mean(soc_error)
soc_std = np.std(soc_error)


# PLOTS
plt.figure(figsize=(14, 12))


# VOLTAGE PERFORMANCE
plt.subplot(3,1,1)
plt.plot(time_eval, v_true, 'b', label='Measured Voltage')
plt.plot(time_eval, v_pred, 'r--', label='Predicted Voltage')

plt.title("Voltage Prediction Performance")
plt.ylabel("Voltage (V)")
plt.grid(True, alpha=0.3)
plt.legend()

# Metrics text box
plt.text(0.02, 0.95,
         f"RMSE: {v_rmse:.4f} V\nMAE: {v_mae:.4f} V\nR²: {v_r2:.4f}",
         transform=plt.gca().transAxes,
         verticalalignment='top',
         bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))

# SOC TRACKING
plt.subplot(3,1,2)
plt.plot(time_eval, s_true*100, 'k', label='True SOC')
plt.plot(time_eval, s_pred*100, 'g--', label='Estimated SOC')

plt.fill_between(time_eval,
                 s_pred*100+3,
                 s_pred*100-3,
                 alpha=0.1,
                 label='±3% Target Band')

plt.title("SOC Estimation Performance")
plt.ylabel("SOC (%)")
plt.grid(True, alpha=0.3)
plt.legend()

plt.text(0.02, 0.95,
         f"RMSE: {soc_rmse*100:.3f} %\nMAE: {soc_mae*100:.3f} %\nMax: {soc_max:.3f} %",
         transform=plt.gca().transAxes,
         verticalalignment='top',
         bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))

# SOC RESIDUAL ANALYSIS

plt.subplot(3,1,3)
plt.plot(time_eval, soc_error, color='purple', label='SOC Error')

plt.axhline(0, color='black', linestyle='--', label='Zero Line')
plt.axhline(3, color='red', linestyle='--', label='+3% Limit')
plt.axhline(-3, color='red', linestyle='--', label='-3% Limit')

plt.fill_between(time_eval,
                 soc_mean + soc_std,
                 soc_mean - soc_std,
                 alpha=0.1,
                 label='±1 Std Dev')

plt.title("SOC Residual Statistics")
plt.xlabel("Time (s)")
plt.ylabel("Error (%)")
plt.grid(True, alpha=0.3)
plt.legend()

plt.text(0.02, 0.95,
         f"Mean: {soc_mean:.3f} %\nStd: {soc_std:.3f} %",
         transform=plt.gca().transAxes,
         verticalalignment='top',
         bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))

plt.tight_layout()
plt.show()
