import glob
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.impute import SimpleImputer
from sklearn.metrics import r2_score
from pandas import DataFrame

import sys
sys.path.insert(0, str(Path(__file__).parents[1]))
from utils.config import MD_RESULTS_DIR

# Constants
k_B = 1.380649e-23  # J/K
e = 1.60218e-19     # J/eV

def calc_diff(msd, t):
    reg = LinearRegression()
    reg.fit(t.reshape(-1,1), msd)
    y_pred = reg.predict(t.reshape(-1,1))
    r2 = r2_score(msd, y_pred)
    print(f"Diffusion coefficient: {reg.coef_[0]/6:.3e} Å²/ps (r²={r2:.3f})")
    return np.float64(reg.coef_[0]/6), y_pred, r2

def calc_act(diffusion_dict):
    temps = sorted(diffusion_dict.keys())
    inv_T = 1 / np.array(temps)
    D_vals = [diffusion_dict[T] for T in temps]
    imp = SimpleImputer(missing_values=np.nan, strategy='mean')
    lnD = np.log(imp.fit_transform(np.array(D_vals).reshape(-1, 1)).astype('float'))

    reg = LinearRegression()
    reg.fit(inv_T.reshape(-1, 1), lnD)
    y_pred = reg.predict(inv_T.reshape(-1, 1))

    residuals = lnD.flatten() - y_pred.flatten()
    n = len(temps)
    s_err = np.sqrt(np.sum(residuals**2) / (n - 2)) / np.sqrt(np.sum((inv_T - np.mean(inv_T))**2))

    slope = reg.coef_[0][0]
    Ea_J = -slope * k_B
    Ea_eV = Ea_J / e
    dEa_J = s_err * k_B
    dEa_eV = dEa_J / e

    return float(Ea_eV), float(dEa_eV), y_pred, temps


# Load MSD data
msd_data = {}
data = {}

for file in glob.glob(str(MD_RESULTS_DIR / "*" / "*" / "*.npz")):
    parts = file.replace("\\", "/").split("/")
    material = parts[-3]
    model = parts[-2]
    filename = parts[-1]
    
    # Extract temperature from filename (last underscore-separated component, minus .npz extension)
    temp = int(filename.replace(".npz", "").split("_")[-1])
    
    arr = np.load(file)
    t, msd = arr["lag_times_ps"], arr["msd_A2"]

    diffusion, diff_line, r2_val = calc_diff(msd, t)
    diff_m = diffusion * 1e-8  # m²/s

    print(f"{material} {model} {temp} K --> {diff_m:.3e} m²/s (r²={r2_val:.3f}); ({len(msd)} pts)")

    # Store MSD plot data
    if material not in msd_data:
        msd_data[material] = {}
    if model not in msd_data[material]:
        msd_data[material][model] = {}
    msd_data[material][model][temp] = {
        't': t, 
        'msd': msd, 
        'diff_line': diff_line, 
        'r2': r2_val,
        'diffusion': diff_m
    }

    # Store results
    if material not in data:
        data[material] = {}
    if model not in data[material]:
        data[material][model] = {}
    data[material][model][temp] = diff_m


# Create individual MSD plots
for material, models_dict in msd_data.items():
    for model, temps_dict in models_dict.items():
        for temp, plot_data in temps_dict.items():
            t = plot_data['t']
            msd = plot_data['msd']
            diff_line = plot_data['diff_line']
            r2 = plot_data['r2']
            diff_m = plot_data['diffusion']
            
            # Individual MSD plot
            f, ax = plt.subplots()
            ax.plot(t, msd, label=f'MSD, D: {diff_m:.3e} m²/s')
            ax.plot(t, diff_line, "--", color="purple", label=f"Fit, (r²={r2:.2f})")
            ax.set_xlabel("Time (ps)")
            ax.set_ylabel("MSD (Å²)")
            ax.legend()
            f.savefig(f"results/figures/msd/{material}_{model}_{temp}.png", dpi=300)
            plt.close()


# Ea plotting
activation = {}
for material, models_dict in data.items():
    predictions = {}
    f2, ax2 = plt.subplots()
    f1, ax1 = plt.subplots()

    for model, values in models_dict.items():
        # Filter by r2 cutoff
        filtered_temps = []
        filtered_values = {}
        for temp, diff_value in values.items():
            if material in msd_data and model in msd_data[material] and temp in msd_data[material][model]:
                r2 = msd_data[material][model][temp]['r2']
                if r2 >= 0.8 and diff_value > 0:
                    filtered_temps.append(temp)
                    filtered_values[temp] = diff_value
                else:
                    print(f"Discarding {material} {model} {temp} (r²={r2:.2f}, D={diff_value:.3e})")
        
        # Only keep datasets with >=2 valid diffusion values
        if len(filtered_values) < 2:
            print(f"Skipping {material} {model} (not enough valid points)")
            continue
        
        values = filtered_values

        Ea, Ea_err, y_pred, T_list = calc_act(values)
        predictions[model] = (Ea, Ea_err)
        inv_T = 1000 / np.array(T_list)
        D_list = [values[T] for T in T_list]
        label = f"{model}: {Ea:.3f}({Ea_err*1000:.0f}) eV"
        ax2.scatter(inv_T, np.log(D_list), label=label, s=60)
        ax2.plot(inv_T, y_pred)

        ax1.scatter(T_list, D_list, label=model, s=60)

    ax2.set_xlabel("1000/T [K⁻¹]", fontsize=18)
    ax2.set_ylabel("ln(D)", fontsize=18)
    ax2.tick_params(labelsize=14)
    for spine in ax2.spines.values():
        spine.set_linewidth(1.5)
    ax2.legend()
    f2.tight_layout()
    f2.savefig(f"results/figures/activation_{material}.png", dpi=300)
    f2.savefig(f"results/figures/activation_{material}.svg")

    ax1.set_xlabel("Temperature [K]", fontsize=18)
    ax1.set_ylabel("Diffusion [m²/s]", fontsize=18)
    ax1.tick_params(labelsize=14)
    for spine in ax1.spines.values():
        spine.set_linewidth(1.5)
    ax1.legend()
    f1.tight_layout()
    f1.savefig(f"results/figures/diffusion_{material}.png", dpi=300)
    f1.savefig(f"results/figures/diffusion_{material}.svg")

    activation[material] = predictions

# Save activation energy
df = DataFrame.from_dict({(i,j): activation[i][j] for i in activation for j in activation[i]}, 
                         columns=["Ea", "dEa"], orient="index")
df.to_excel("results/activation.xlsx", sheet_name="MD_Activation")

# Save diffusion
diffusion_dict = {}
for material in data:
    for model in data[material]:
        if data[material][model]:  # only add non-empty
            diffusion_dict[(material, model)] = data[material][model]

df_diff = DataFrame.from_dict(diffusion_dict, orient="index").sort_index(axis=1)
df_diff.to_excel("results/diffusion.xlsx", sheet_name="MD_Diffusion")
