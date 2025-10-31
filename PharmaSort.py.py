#!/usr/bin/env python
# coding: utf-8

# In[2]:


import glob
import pandas as pd
import warnings
import numpy as np
import os


# In[3]:


from mfc_aspen import MFC 
from aspen import AspenPlus as ap
import warnings
warnings.filterwarnings("ignore")


# In[4]:


aspen_model_paths = glob.glob('PharmaSort APAP\*\*.bkp')# path to aspen model 


# In[7]:


#Insert names accordingly based on file paths
aspen_model_names = ['Acetaminophen','Acetic Acid','Acetic Anhydride', 'Acetone','Benzene' ,'Hydrogen SMR','Iso propanol','Methane','Nitro benzene', 'Para aminophenol','Propene']


# In[8]:


from aspen import AspenPlus as ap
import pandas as pd

class MFC:

    @staticmethod
    def extract_waste_compositions(aspen_model_paths, aspen_model_names):
        all_waste_data = []

        for model_path, model_name in zip(aspen_model_paths, aspen_model_names):
            print(f"\n🔍 Checking WASTE streams in model: {model_name}")

            a = ap()
            a.simulate(model_path)

            # Get all streams
            in_streams = a.get_in_streams()
            out_streams = a.get_out_streams()
            all_streams = in_streams + out_streams

            # Find waste streams
            waste_streams = [s for s in all_streams if "WASTE" in s.upper()]
            if not waste_streams:
                print(f"⚠️ No WASTE streams found in {model_name}")
                a.quit()
                continue

            # Extract compositions for each waste stream
            for stream in waste_streams:
                try:
                    mf = a.get_massfrac(stream).round(5)
                    mf = mf[mf["Mass_Frac"] > 0]  # filter nonzero fractions
                    mf["Model"] = model_name
                    mf["Stream"] = stream
                    all_waste_data.append(mf)
                    print(f"✅ Added composition for {stream} ({model_name})")
                except Exception as e:
                    print(f"❌ Could not read {stream} in {model_name}: {e}")

            a.quit()

        # Combine all into one DataFrame
        if not all_waste_data:
            print("⚠️ No WASTE data found in any model.")
            return None

        combined_df = pd.concat(all_waste_data, ignore_index=True)
        combined_df = combined_df[["Model", "Stream", "Component_Name", "Mass_Frac"]]

        # Save to CSV
        combined_df.to_csv("All_WASTE_Compositions.csv", index=False)
        print("💾 Saved combined WASTE stream compositions → All_WASTE_Compositions.csv")

        return combined_df


# ---------------------------------------------------------------
# DRIVER SCRIPT
# ---------------------------------------------------------------

# Example usage:
# aspen_model_paths = ["C:\\Users\\...\\IPA-Water.bkp", "C:\\Users\\...\\Acetone-Water.bkp"]
# aspen_model_names = ["IPA_Water", "Acetone_Water"]

MFC.extract_waste_compositions(aspen_model_paths, aspen_model_names)


# In[ ]:


# run_selection_from_streams.py
# Input: a CSV with columns [Model, Stream, Component_Name, Mass_Frac]
# Output: per-(Model,Stream) Bayesian tech ranking CSVs (+ optional plots)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from rdkit import Chem
from rdkit.Chem import Descriptors
from dataclasses import dataclass
import os

# ----------------------
# CONFIG
# ----------------------
INPUT_CSV   = "All_WASTE_Compositions.csv"     # <-- put your table in this file
OUT_DIR     = "selection_results"
N_SAMPLES   = 2000
GAMMA       = 12        # Dirichlet concentration (tightness around mean)
RNG_SEED    = 7
PAIR_PLOTS  = [("CapEx","CO2"),("Energy","OpEx"),("Energy","WasteReduction"),("CO2","WasteReduction")]

os.makedirs(OUT_DIR, exist_ok=True)

# ----------------------
# Utilities
# ----------------------
def sigmoid(z): return 1.0/(1.0+np.exp(-z))

# Map Aspen-ish IDs to approximate SMILES for descriptors (extend as you learn IDs)
SMILES = {
    # Common
    "WATER": "O", "H2O": "O",
    "CO2": "O=C=O", "METHANE": "C", "ETHANE":"CC", "PROPANE":"CCC",
    "HYDROGEN":"[H][H]", "H2":"[H][H]",
    # Organics / solvents
    "ACETONE":"CC(=O)C", "IPA":"CC(O)C",
    "BENZENE":"c1ccccc1",
    "KETENE":"C=C=O",
    "ACETIC":"CC(=O)O", "ACETACID":"CC(=O)O",  # acetic acid
    "ANILINE":"Nc1ccccc1", "H-ANILIN":"Nc1ccccc1",
    "NAPHT-01":"c1ccc2ccccc2c1",               # naphthalene proxy
    "BENZE-01":"c1ccccc1",                     # benzene proxy
    "NITRI-01":"O=[N+]([O-])c1ccccc1",         # nitrobenzene proxy
    "SULFU-01":"O=S(=O)(O)O",                  # sulfuric acid proxy
    "DEA":"OCCN(CCO)CCO",                      # diethanolamine
    "METHY-01":"CO"                            # methanol proxy
    # Add more as needed...
}

AROM_FLAGS = {"c","n","o"}  # crude aromatic detection is done via RDKit anyway

def rdkit_desc(smiles: str):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return dict(LogP=np.nan, MW=np.nan, a=0, h=0)
    logp = Descriptors.MolLogP(mol)
    mw   = Descriptors.MolWt(mol)
    aromatic = int(any(a.GetIsAromatic() for a in mol.GetAtoms()))
    halogen  = int(any(a.GetSymbol() in {"F","Cl","Br","I"} for a in mol.GetAtoms()))
    return dict(LogP=logp, MW=mw, a=aromatic, h=halogen)

def component_descriptors(comp_name: str):
    key = comp_name.upper()
    smi = SMILES.get(key)
    if not smi:
        # unknown → neutral fallback
        return dict(LogP=0.0, MW=60.0, a=0, h=0)
    return rdkit_desc(smi)

def mixture_descriptors(df_group: pd.DataFrame):
    """Mass-weighted mixture descriptors; crude Tb and BIOWIN proxies."""
    w = df_group["Mass_Frac"].to_numpy()
    w = w / (w.sum() + 1e-12)
    LogP = MW = a = h = 0.0
    for i, (_, r) in enumerate(df_group.iterrows()):
        d = component_descriptors(str(r["Component_Name"]))
        LogP += w[i]*float(d["LogP"])
        MW   += w[i]*float(d["MW"])
        a    += w[i]*int(d["a"])
        h    += w[i]*int(d["h"])
    # crude Tb proxy from MW (replace with databank if available)
    Tb = 100.0 + 0.25*(MW - 60.0)
    # BIOWIN proxy: organics ~ degradable, inorganics ~ 0
    # Heuristic: if component has a SMILES with carbon, give base 0.7; salts/acids low
    organic_frac = 0.0
    for i, (_, r) in enumerate(df_group.iterrows()):
        key = str(r["Component_Name"]).upper()
        smi = SMILES.get(key, "")
        if any(ch in smi for ch in ["C","c"]):
            organic_frac += w[i]
    biowin = 0.2 + 0.7*min(1.0, organic_frac/0.5)  # 0.2..0.9
    return dict(LogP=LogP, MW=MW, a=a, h=h, Tb=Tb, b=biowin)

# ----------------------
# Likelihoods -> posterior mean weights
# ----------------------
def expected_likelihoods(feat, L_alpha=0.85, T_th=120.0, kv=0.06,
                         alpha_capex=0.6, alpha_opex=0.7, w_b=0.4, w_eta=0.6):
    LogP, a, h, MW, Tb, b = feat["LogP"], feat["a"], feat["h"], feat["MW"], feat["Tb"], feat["b"]
    L_E   = 0.7*L_alpha + 0.3*sigmoid(LogP - 2.5)
    L_CO2 = min((a+h)/6.0, 1.0) + 0.5*L_E
    L_CX  = alpha_capex*min(MW/200.0,1.0) + (1-alpha_capex)*sigmoid(kv*(Tb - T_th))
    L_OX  = alpha_opex*L_E + (1-alpha_opex)*sigmoid(LogP - 2.0)
    eta   = sigmoid(0.06*(Tb - 100.0)) + 0.2*(a+h)
    eta   = np.clip(eta, 0.0, 1.0)
    L_WR  = w_b*(1-b) + w_eta*eta
    d = {"Energy":L_E,"CO2":L_CO2,"CapEx":L_CX,"OpEx":L_OX,"WasteReduction":L_WR}
    s = sum(d.values())
    return {k:v/s for k,v in d.items()}

def sample_dirichlet(mean_w: dict, n=N_SAMPLES, gamma=GAMMA, seed=RNG_SEED):
    names = ["Energy","CO2","CapEx","OpEx","WasteReduction"]
    wbar  = np.array([mean_w[k] for k in names], dtype=float)
    wbar /= wbar.sum()
    rng = np.random.default_rng(seed)
    W = rng.dirichlet(alpha=gamma*wbar, size=n)  # (n,5)
    return names, W

# ----------------------
# Technology eligibility & KPI surrogates
# ----------------------
@dataclass
class Tech:
    name: str
    cls:  str

def eligible_techs(df_group: pd.DataFrame):
    comp_upper = df_group["Component_Name"].str.upper().tolist()
    w = df_group.set_index(df_group["Component_Name"].str.upper())["Mass_Frac"]
    f_salts = float(w.get("NACL", 0.0) + w.get("SULFU-01", 0.0))  # crude salts/acid proxy
    f_acid  = float(w.get("ACETIC",0.0) + w.get("ACETACID",0.0) + w.get("SULFU-01",0.0))
    f_ipa   = float(w.get("IPA",0.0))
    f_water = float(w.get("WATER",0.0) + w.get("H2O",0.0))

    techs = []
    # Distillation if salts low
    if f_salts < 0.05:
        techs.append(Tech("Distillation","thermal"))
    # Pervaporation if IPA-water present
    if f_ipa > 0.01 and f_water > 0.3:
        techs.append(Tech("Pervaporation","membrane"))
    # Desal/acid polish membrane if salts/acids material
    if f_salts > 0.005 or f_acid > 0.01:
        techs.append(Tech("Membrane_Desal_AcidPolish","membrane"))
    # LLE if heavy aromatics fraction (benzene/naphthalene proxies)
    f_arom = float(w.get("BENZENE",0.0) + w.get("BENZE-01",0.0) + w.get("NAPHT-01",0.0) + w.get("NITRI-01",0.0))
    if f_arom > 0.05:
        techs.append(Tech("LLE_AromaticSelective","extraction"))
    # Fallback if none
    if not techs:
        techs = [Tech("Distillation","thermal")]
    return techs

def kpi_surrogate(feat: dict, tech: Tech):
    LogP, MW, Tb, a, h, b = feat["LogP"], feat["MW"], feat["Tb"], feat["a"], feat["h"], feat["b"]
    L_E = 0.7*0.85 + 0.3*sigmoid(LogP - 2.5)
    base_energy = 10000  # scale for a nominal stream; relative comparisons hold

    if tech.name == "Distillation":
        energy = base_energy * (1.05 + 1.00*L_E)
        co2    = 0.08 * energy
        capex  = 1.05
        opex   = 650 + 60*L_E
        wr     = 0.90
    elif tech.name == "Pervaporation":
        energy = base_energy * (0.65 + 0.70*L_E)
        co2    = 0.07 * energy
        capex  = 0.95
        opex   = 600 + 40*L_E
        wr     = 0.92
    elif tech.name == "Membrane_Desal_AcidPolish":
        energy = base_energy * 0.45
        co2    = 0.06 * energy
        capex  = 0.85
        opex   = 560
        wr     = 0.88
    elif tech.name == "LLE_AromaticSelective":
        energy = base_energy * 0.50
        co2    = 0.055 * energy
        capex  = 0.90
        opex   = 570
        wr     = 0.86
    else:
        energy = base_energy
        co2, capex, opex, wr = 0.08*energy, 1.00, 600, 0.85

    return dict(Energy=energy, CO2=co2, CapEx=capex, OpEx=opex, WasteReduction=wr)

# ----------------------
# Normalization + posterior scoring
# ----------------------
def to_losses(df_kpi: pd.DataFrame):
    d = df_kpi.copy()
    for c in ["Energy","CO2","CapEx","OpEx"]:
        lo, hi = d[c].min(), d[c].max()
        d[c+"_loss"] = (d[c]-lo)/(hi-lo+1e-12)
    lo, hi = d["WasteReduction"].min(), d["WasteReduction"].max()
    d["WasteReduction_loss"] = 1 - (d["WasteReduction"]-lo)/(hi-lo+1e-12)
    loss_cols = [x+"_loss" for x in ["Energy","CO2","CapEx","OpEx","WasteReduction"]]
    return d, loss_cols

def score(df_kpi_loss: pd.DataFrame, loss_cols, mean_w, n=N_SAMPLES, gamma=GAMMA, seed=RNG_SEED):
    _, W = sample_dirichlet(mean_w, n=n, gamma=gamma, seed=seed)  # (n,5)
    F = df_kpi_loss[loss_cols].values  # (T,5)
    S = W @ F.T                        # (n,T)
    mean = S.mean(axis=0)
    p5,p50,p95 = np.percentile(S,[5,50,95],axis=0)
    prob_best = (S.argmin(axis=1)[:,None] == np.arange(S.shape[1])).mean(axis=0)
    return S, dict(mean=mean, p5=p5, p50=p50, p95=p95, prob_best=prob_best)

# ----------------------
# Plots (optional)
# ----------------------
def plot_pareto_pairs(df_kpi: pd.DataFrame, out_prefix: str):
    tech = df_kpi["Tech"].values
    for xk, yk in PAIR_PLOTS:
        x = df_kpi[xk].values
        y = df_kpi[yk].values if yk!="WasteReduction" else (1 - df_kpi["WasteReduction"].values)
        plt.figure(figsize=(7,5))
        plt.scatter(x, y, s=120, alpha=0.85)
        for i,t in enumerate(tech):
            plt.text(x[i]*1.01, y[i]*1.01, t, fontsize=9)
        plt.xlabel(xk + (" [lower=better]" if xk!="WasteReduction" else " [higher=better]"))
        plt.ylabel(yk if yk!="WasteReduction" else "1 - WasteReduction (loss)")
        plt.title(f"Pareto: {xk} vs {yk}")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f"{out_prefix}_pareto_{xk}_vs_{yk}.png", dpi=300)
        plt.close()

def plot_uncertainty(df_kpi: pd.DataFrame, stats: dict, out_png: str, title_suffix=""):
    techs = df_kpi["Tech"].values
    xpos = np.arange(1, len(techs)+1)
    plt.figure(figsize=(8,5))
    plt.vlines(xpos, stats["p5"], stats["p95"], linestyles='-', lw=4, alpha=0.7, label='5–95% interval')
    plt.scatter(xpos, stats["p50"], marker='o', s=70, label='Median')
    plt.scatter(xpos, stats["mean"], marker='s', s=70, label='Mean')
    plt.xticks(xpos, techs)
    plt.ylabel("Composite score (lower = better)")
    ttl = "Posterior Uncertainty of Technology Scores"
    if title_suffix:
        ttl += f"\n{title_suffix}"
    plt.title(ttl)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_png, dpi=300)
    plt.close()

# ----------------------
# Orchestrator
# ----------------------
def process_group(df_group: pd.DataFrame, tag: str, make_plots=True):
    # normalize mass fractions within group
    s = df_group["Mass_Frac"].sum()
    if s > 0:
        df_group = df_group.copy()
        df_group["Mass_Frac"] = df_group["Mass_Frac"]/s

    feat   = mixture_descriptors(df_group)
    mean_w = expected_likelihoods(feat)
    techs  = eligible_techs(df_group)

    # KPI table via surrogate (replace with Aspen KPIs as desired)
    rows = []
    for t in techs:
        kp = kpi_surrogate(feat, t)
        rows.append({"Tech":t.name, **kp})
    kpi_df = pd.DataFrame(rows)

    # normalize -> losses; posterior scoring
    kpi_loss_df, loss_cols = to_losses(kpi_df)
    S, stats = score(kpi_loss_df, loss_cols, mean_w, n=N_SAMPLES, gamma=GAMMA, seed=RNG_SEED)

    # assemble result table
    out = kpi_df.copy()
    out["MeanScore"] = stats["mean"]
    out["P50"]       = stats["p50"]
    out["P05"]       = stats["p5"]
    out["P95"]       = stats["p95"]
    out["ProbBest"]  = stats["prob_best"]
    out = out.sort_values("MeanScore").reset_index(drop=True)

    # save
    csv_path = os.path.join(OUT_DIR, f"ranking_{tag}.csv")
    out.to_csv(csv_path, index=False)

    # plots
    if make_plots:
        prefix = os.path.join(OUT_DIR, f"{tag}")
        plot_pareto_pairs(kpi_df, prefix)
        plot_uncertainty(kpi_df, stats, f"{prefix}_uncertainty.png", title_suffix=tag)

    return out, mean_w

def main():
    df = pd.read_csv(INPUT_CSV)
    need_cols = {"Model","Stream","Component_Name","Mass_Frac"}
    assert need_cols.issubset(df.columns), f"CSV must contain: {need_cols}"

    results = []
    for (model, stream), grp in df.groupby(["Model","Stream"], dropna=False):
        tag = f"{str(model).replace(' ','_')}_{str(stream)}"
        print(f"Processing: {tag}")
        out, mean_w = process_group(grp, tag=tag, make_plots=True)
        best = out.iloc[0]
        results.append({
            "Model": model, "Stream": stream,
            "BestTech": best["Tech"],
            "BestMeanScore": best["MeanScore"],
            "ProbBest": best["ProbBest"],
            "PosteriorWeights": mean_w
        })

    summary = pd.DataFrame(results)
    summary.to_csv(os.path.join(OUT_DIR, "summary_all_streams.csv"), index=False)
    print("\n=== Summary ===")
    print(summary.to_string(index=False))
    print(f"\nSaved results in: {OUT_DIR}/")

if __name__ == "__main__":
    main()


# In[ ]:




