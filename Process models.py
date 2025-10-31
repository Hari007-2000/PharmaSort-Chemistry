#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# process_models.py
# Fast, closed-form KPI surrogates for recovery technologies
# Units: feed flow in kg/h, composition mass fractions
# Outputs normalized per metric ton of FEED (1000 kg)

from dataclasses import dataclass
import math

# ---------------------------
# Global defaults / economics
# ---------------------------
EF_GRID = 0.45            # kg CO2 per kWh (adjust to your grid)
USD_per_kWh = 0.10        # electricity price
CAPEX_CR_F = 0.15         # capital recovery factor (i=10%, n=10y → ~0.162; tune)
ANNUAL_H = 8000           # operating hours per year
USD_per_Stage = 8000      # distillation internals cost per stage (very rough)
USD_per_m2_membrane = 300 # membrane module cost
USD_per_m3_mixer = 15000  # mixer-settler capital proxy per m3
USD_per_m2_HX = 600       # condenser/reboiler area cost proxy
SOLVENT_COST_USD_kg = 1.5 # LLE solvent makeup cost (if used)

# Helper: per-ton normalization
def per_ton(value_per_hour, feed_kgph):
    """Convert 'per hour' intensive quantity tied to feed to 'per ton of feed' basis."""
    tons_per_hour = feed_kgph / 1000.0
    return value_per_hour / max(tons_per_hour, 1e-9)

# ---------------------------
# Distillation (binary/semi)
# ---------------------------
@dataclass
class DistillParams:
    # thermo / design proxies
    rel_vol: float = 1.8       # relative volatility (effective)
    q_reflux: float = 1.2      # operating reflux ratio multiple of Rmin
    latent_kJkg_light: float = 800  # effective latent heat (kJ/kg distillate)
    eta_util: float = 0.85     # utilities effectiveness (HX, losses)
    xD: float = 0.95           # target distillate purity (light key in D)
    xB: float = 0.05           # bottoms light-key
    stages_min: int = 12       # baseline trays
    fouling_factor: float = 1.0  # >1 if dirty feeds

def distillation_kpis(feed_kgph: float,
                      mass_frac_light: float,
                      params: DistillParams = DistillParams()):
    """
    Simple FUG proxy:
    - Rmin via Underwood (binary short-cut → approx Rmin ≈ (xD/xF - 1)/(alpha-1) clipped)
    - Stages via Gilliland shape (fixed here)
    - Duty ≈ (R+1)*D*latent / eta
    Waste reduction: fraction of target recovered to distillate.
    """
    xF = max(min(mass_frac_light, 0.999), 1e-3)
    alpha = max(params.rel_vol, 1.05)
    # Underwood-ish Rmin (safe proxy)
    Rmin = max((params.xD/xF - 1.0) / max(alpha - 1.0, 1e-3), 0.5)
    R = params.q_reflux * Rmin

    # Distillate rate (mass basis) assuming sharp split of light in D:
    D_kgph = feed_kgph * ( (params.xD - params.xB) / max(xF - params.xB, 1e-6) ) * xF
    D_kgph = max(min(D_kgph, feed_kgph), 0.0)

    # Thermal duty (kW): (R+1)*D * latent / (3600 s) / eta  (kJ/kg to kW)
    Q_kW = ((R + 1.0) * D_kgph * params.latent_kJkg_light) / (3600.0 * params.eta_util)
    Q_kWh_per_ton = per_ton(Q_kW, feed_kgph)  # kWh/ton feed, since kW per h per ton

    # CO2
    CO2_kg_per_ton = Q_kWh_per_ton * EF_GRID

    # CapEx (very rough): trays + exchangers + diameter proxy via flow
    stages = params.stages_min
    capex_base = stages * USD_per_Stage + 50_000 + 0.05 * feed_kgph  # tiny flow factor
    capex_annual = CAPEX_CR_F * capex_base * params.fouling_factor

    # OpEx: energy + utilities + minor solvent + maintenance
    opex_energy = Q_kWh_per_ton * USD_per_kWh  # USD/ton
    opex_maint = 0.05 * (capex_annual / (ANNUAL_H/1.0))  # USD/h → per ton converts below
    # Convert opex_maint to USD/ton:
    opex_maint_per_ton = per_ton(opex_maint, feed_kgph)
    OpEx_USD_per_ton = opex_energy + opex_maint_per_ton
    OpEx_USD_per_yr  = OpEx_USD_per_ton * (feed_kgph/1000.0) * ANNUAL_H

    # Waste reduction: assume light impurity removed from bottoms & heavies from D (proxy)
    WR = 100.0 * (1.0 - (1.0 - xF) * (1.0 - params.xD))  # simplistic: higher with purer D

    return dict(Energy=Q_kWh_per_ton,
                CO2=CO2_kg_per_ton,
                CapEx=capex_annual,
                OpEx=OpEx_USD_per_yr,
                WasteReduction=WR)

# ---------------------------
# Pervaporation (alcohol-water style)
# ---------------------------
@dataclass
class PVParams:
    J0_kg_m2_h: float = 0.5     # baseline flux at reference driving force
    beta: float = 1.0           # proportional factor to activity diff
    thickness_um: float = 2.0   # not used explicitly here (absorbed in J0)
    permeate_frac: float = 0.2  # fraction of feed drawn through membrane
    Hvap_kJkg: float = 2250     # latent heat for condensation of permeate (kJ/kg)
    vac_pump_kWh_per_kg: float = 0.02  # vacuum/aux energy
    membrane_life_years: float = 2.0

def pervaporation_kpis(feed_kgph: float,
                       x_alcohol: float, x_water: float,
                       params: PVParams = PVParams()):
    """
    Solution-diffusion proxy:
    J ≈ J0 * beta * Δa; use Δa ~ x_alcohol + 0.3*x_water (polar bias).
    Size area from permeate rate; energy ~ Hvap*permeate + vac pump.
    """
    delta_a = max(1e-3, x_alcohol + 0.3 * x_water)
    J = params.J0_kg_m2_h * params.beta * delta_a  # kg/m2-h

    m_perm = params.permeate_frac * feed_kgph  # kg/h
    area_m2 = m_perm / max(J, 1e-9)

    # Energy (kWh/h): condense permeate + vacuum work
    E_kW = (m_perm * params.Hvap_kJkg) / 3600.0 + m_perm * params.vac_pump_kWh_per_kg
    E_kWh_per_ton = per_ton(E_kW, feed_kgph)

    CO2_kg_per_ton = E_kWh_per_ton * EF_GRID

    # CapEx: membrane modules
    capex_modules = area_m2 * USD_per_m2_membrane
    capex_annual = CAPEX_CR_F * capex_modules

    # OpEx: energy + membrane replacement
    opex_energy_per_ton = E_kWh_per_ton * USD_per_kWh
    mem_replace_per_year = capex_modules / params.membrane_life_years
    OpEx_USD_per_yr = opex_energy_per_ton * (feed_kgph/1000.0) * ANNUAL_H + mem_replace_per_year

    # Waste reduction: remove permeated contaminant (assume alcohol preferential)
    WR = 100.0 * (params.permeate_frac * (x_alcohol + 0.2*x_water))

    return dict(Energy=E_kWh_per_ton,
                CO2=CO2_kg_per_ton,
                CapEx=capex_annual,
                OpEx=OpEx_USD_per_yr,
                WasteReduction=WR)

# ---------------------------
# Membrane Desal/Acid Polish (RO/NF proxy)
# ---------------------------
@dataclass
class ROParams:
    A_L_m2_bar_h: float = 1.5   # water permeability (L/m2-h-bar)
    deltaP_bar: float = 10.0    # transmembrane pressure
    osmotic_bar: float = 3.0    # osmotic pressure (effective)
    recovery: float = 0.5       # water recovery
    pump_eff: float = 0.7
    membrane_life_years: float = 3.0

def ro_desal_kpis(feed_kgph: float,
                  x_salts_acids: float,
                  params: ROParams = ROParams()):
    """
    RO specific energy ~ ΔP * permeate_vol / (η * ρ), scaled to feed.
    Flux Jw = A*(ΔP - Δπ). Area from recovery. Capex from area.
    """
    rho = 1000.0  # kg/m3
    # Permeate volumetric flow (m3/h)
    perm_m3ph = params.recovery * (feed_kgph / rho)
    # Flux (L/m2-h) → m3/m2-h
    Jw_m3_m2_h = max((params.A_L_m2_bar_h * 1e-3) * max(params.deltaP_bar - params.osmotic_bar, 0.1), 1e-6)
    area_m2 = perm_m3ph / Jw_m3_m2_h

    # Pump power kW ~ ΔP (Pa) * Q (m3/s) / η
    deltaP_Pa = params.deltaP_bar * 1e5
    Q_m3s = perm_m3ph / 3600.0
    P_kW = (deltaP_Pa * Q_m3s) / (params.pump_eff * 1000.0)  # 1000 → kW

    E_kWh_per_ton = per_ton(P_kW, feed_kgph)
    CO2_kg_per_ton = E_kWh_per_ton * EF_GRID

    capex_modules = area_m2 * USD_per_m2_membrane
    capex_annual = CAPEX_CR_F * capex_modules

    # OpEx: energy + membrane replacement
    opex_energy_per_ton = E_kWh_per_ton * USD_per_kWh
    mem_replace_per_year = capex_modules / params.membrane_life_years
    OpEx_USD_per_yr = opex_energy_per_ton * (feed_kgph/1000.0) * ANNUAL_H + mem_replace_per_year

    # Waste reduction: remove salts/acids to permeate (or retain & purge): proxy
    WR = 100.0 * (params.recovery * max(0.1, 1.0 - 3.0*x_salts_acids))

    return dict(Energy=E_kWh_per_ton,
                CO2=CO2_kg_per_ton,
                CapEx=capex_annual,
                OpEx=OpEx_USD_per_yr,
                WasteReduction=WR)

# ---------------------------
# LLE for Aromatics (single solvent, N stages)
# ---------------------------
@dataclass
class LLEParams:
    Kd: float = 3.0             # distribution coefficient (aromatic to solvent)
    E_overF: float = 1.0        # extraction factor E/F per stage (solvent/feed mass ratio)
    stages: int = 2
    mix_power_kW_per_m3: float = 1.0
    unit_volume_m3: float = 2.0
    solvent_loss_frac: float = 0.01  # fraction of solvent lost & made up
    optional_solvent_recovery_kWh_per_ton: float = 50.0

def lle_aromatic_kpis(feed_kgph: float,
                      x_aromatic: float,
                      params: LLEParams = LLEParams()):
    """
    Stagewise removal for solute S with E/F and Kd:
    Fraction remaining after one stage ≈ 1 / (1 + Kd * (E/F)).
    After N stages: (remaining) = [1 / (1 + Kd*E/F)]^N.
    """
    remain = (1.0 / (1.0 + params.Kd * params.E_overF)) ** params.stages
    removed = 1.0 - remain
    WR = 100.0 * min(1.0, removed * x_aromatic * 3.0)  # scale by aromatic fraction

    # Mixing energy + (optional) solvent recovery (e.g., small polishing distillation)
    P_kW = params.mix_power_kW_per_m3 * params.unit_volume_m3
    E_kWh_per_ton = per_ton(P_kW, feed_kgph) + params.optional_solvent_recovery_kWh_per_ton

    CO2_kg_per_ton = E_kWh_per_ton * EF_GRID

    # CapEx: mixer-settlers + small HX
    capex_base = USD_per_m3_mixer * params.unit_volume_m3 + 20_000
    capex_annual = CAPEX_CR_F * capex_base

    # OpEx: energy + solvent makeup
    opex_energy_per_ton = E_kWh_per_ton * USD_per_kWh
    solvent_makeup_kgph = params.solvent_loss_frac * params.E_overF * feed_kgph
    solvent_makeup_per_ton = per_ton(solvent_makeup_kgph, feed_kgph)
    opex_solvent = solvent_makeup_per_ton * SOLVENT_COST_USD_kg
    OpEx_USD_per_yr = (opex_energy_per_ton + opex_solvent) * (feed_kgph/1000.0) * ANNUAL_H

    return dict(Energy=E_kWh_per_ton,
                CO2=CO2_kg_per_ton,
                CapEx=capex_annual,
                OpEx=OpEx_USD_per_yr,
                WasteReduction=WR)

