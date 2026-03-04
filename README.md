# TERM PROJECT
Developed an automated workflow to estimate cell open-circuit voltage (OCV) vs SOC, equivalent- circuit model (ECM) parameter identification, and a robust SOC estimator from charge/discharge cycling and impedance data.

## 🛠 Installation
1. Clone the repo: `git clone https://github.com/yourusername/battery-project.git`
2. Install dependencies: `pip install -r requirements.txt`

## Data Directory Setup
The script is designed to handle local file paths rather than Google Drive/Colab uploads.Create a folder named data/ in the root directory.Download the NASA Prognostics Data (e.g., Battery B0005).Place the CSV file in the data/ folder.

Important: In main.py, update the DATA_PATH variable to match your filename:DATA_PATH = os.path.join('data', 'your_filename.csv')
   
## Execution Steps 
1: Data Cleaning: Run the script to generate clean_battery_data.csv. This removes gaps in the voltage/current traces.
2: Parameter Identification: The script will run a Genetic Algorithm to find the optimal $R_0, R_1, C_1, R_2, C_2$ values.
3: AEKF Estimation: The Kalman Filter will process the current/voltage data to estimate SOC in real-time.
4: Output: Visualizations will be saved to the results/ folder as .png files
