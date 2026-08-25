import time
import numpy as np
import heatpump2
from pyomo.environ import *

start_time = time.time()
model = ConcreteModel()

# --------------------------
# Sets
# --------------------------
model.SI = Set(initialize=[i for i in heatpump2.SI])  # hot streams
model.SJ = Set(initialize=[j for j in heatpump2.SJ])  # cold streams
model.SK = Set(initialize=[k for k in heatpump2.SK])  # stages 0..NK-1
model.ST = Set(initialize=[k for k in heatpump2.ST])  # 0..NK-2
model.SP = Set(initialize=[p for p in heatpump2.SP])  # heat pumps

# Define index SIJ as the pair (i,j) as long as thin[i] >= tcin[j] + EMAT.
def _SIJ_init(model):
    return ((i, j) for i in model.SI for j in model.SJ
            if (heatpump2.thin[i] - heatpump2.tcin[j] >= heatpump2.EMAT))

model.SIJ = Set(dimen=2, initialize=_SIJ_init)

# --------------------------
# Variables
# --------------------------
# Bounds of the intermediate temperature of the process stream in the condenser.
def tca_bound(model, j, p, k):
    return (heatpump2.tcin[j], heatpump2.tcout[j])

# Bounds of the exchanger duty q in the heat exchanger network.
def q_bound(model, i, j, k):
    return (0, min(heatpump2.ech[i], heatpump2.ecc[j]))

# Exchanger duty between hot stream i and cold stream j at stage k.
model.q = Var(model.SI, model.SJ, model.SK, within=NonNegativeReals, bounds=q_bound)

# Bounds of qhph: hot stream matching with the hot (evaporator) branch of the heat pump.
def qhph_bound(model, i, p, k):
    return (0, heatpump2.ech[i])

# Evaporator duty of the match between hot stream i and heat pump p at stage k.
model.qhph = Var(model.SI, model.SP, model.SK, within=NonNegativeReals, bounds=qhph_bound)

# Bounds of qhpc: cold stream matching with the cold (condenser) branch of the heat pump.
def qhpc_bound(model, j, p, k):
    return (0, heatpump2.ecc[j])

# Condenser duty of the match between cold stream j and heat pump p at stage k.
model.qhpc = Var(model.SJ, model.SP, model.SK, within=NonNegativeReals, bounds=qhpc_bound)

# Heat-pump fluid mass flowrate in the match between hot stream i and heat pump p (evaporator side).
model.mhph = Var(model.SI, model.SP, model.SK, within=NonNegativeReals, bounds=(0, 100))
# Heat-pump fluid mass flowrate in the match between cold stream j and heat pump p (condenser side).
model.mhpc = Var(model.SJ, model.SP, model.SK, within=NonNegativeReals, bounds=(0, 100))
# Total mass flowrate of the heat-pump working fluid of pump p.
model.mhp = Var(model.SP, within=NonNegativeReals, bounds=(0, 100))

# Bounds of the cooler duty qc of hot stream i.
def qc_bound(model, i):
    return (0, heatpump2.ech[i])

# Bounds of the heater duty qh of cold stream j.
def qh_bound(model, j):
    return (0, heatpump2.ecc[j])

# Heat removed from hot stream i by its cooler.
model.qc = Var(model.SI, within=NonNegativeReals, bounds=qc_bound)
# Heat supplied to cold stream j by its heater.
model.qh = Var(model.SJ, within=NonNegativeReals, bounds=qh_bound)

# Bounds of the hot-stream temperature at each stage.
def th_bound(model, i, k):
    return (heatpump2.thout[i], heatpump2.thin[i])

# Bounds of the cold-stream temperature at each stage.
def tc_bound(model, j, k):
    return (heatpump2.tcin[j], heatpump2.tcout[j])

# Temperature of hot stream i at the hot end of stage k.
model.th = Var(model.SI, model.SK, within=NonNegativeReals, bounds=th_bound)
# Temperature of cold stream j at the hot end of stage k.
model.tc = Var(model.SJ, model.SK, within=NonNegativeReals, bounds=tc_bound)

# Bounds of the compressor power of heat pump p.
def w_bound(model, p):
    return (0, heatpump2.wmax[p])

# Mechanical power input to the compressor of heat pump p.
model.w = Var(model.SP, within=NonNegativeReals, bounds=w_bound)

# Bounds of the exchanger approach temperature dt(i,j,k).
def dt_bound(model, i, j, k):
    return (heatpump2.EMAT, heatpump2.gamma[i, j])

# Approach temperature of the process-process exchanger (i,j,k).
model.dt = Var(model.SI, model.SJ, model.SK, within=NonNegativeReals, bounds=dt_bound)

# Bounds of the cooler approach temperature.
def dtcu_bound(model, i):
    return (heatpump2.EMAT, heatpump2.gammacu[i])

# Bounds of the heater approach temperature.
def dthu_bound(model, j):
    return (heatpump2.EMAT, heatpump2.gammahu[j])

# Bounds of the evaporator approach temperature.
def dthph_bound(model, i, p, k):
    return (heatpump2.EMAT, heatpump2.gammae[i, p, k])

# Bounds of the condenser approach temperature.
def dthpc_bound(model, j, k, p):
    return (heatpump2.EMAT, heatpump2.gammajk[j, k])

# Approach temperature at the cold end of the cooler of hot stream i.
model.dtcu = Var(model.SI, within=NonNegativeReals, bounds=dtcu_bound)
# Approach temperature at the cold end of the heater of cold stream j.
model.dthu = Var(model.SJ, within=NonNegativeReals, bounds=dthu_bound)
# Approach temperature of the evaporator match (i,p,k).
model.dthph = Var(model.SI, model.SP, model.SK, within=NonNegativeReals, bounds=dthph_bound)
# Approach temperature at the hot end of the condenser match (j,p,k).
model.dthpc = Var(model.SJ, model.SK, model.SP, within=NonNegativeReals, bounds=dthpc_bound)
# Approach temperature of the sensible (desuperheating) part of the condenser match.
model.dthpcs = Var(model.SJ, model.SK, model.SP, within=NonNegativeReals, bounds=dthpc_bound)

# Binary existence variables.
model.z = Var(model.SI, model.SJ, model.SK, within=Binary)      # process-process exchanger
model.zhph = Var(model.SI, model.SP, model.SK, within=Binary)   # evaporator match
model.zhpc = Var(model.SJ, model.SP, model.SK, within=Binary)   # condenser match
model.zhu = Var(model.SJ, within=Binary)                        # heater
model.zcu = Var(model.SI, within=Binary)                        # cooler

# Intermediate temperature of cold stream j inside the condenser of pump p at stage k.
model.tca = Var(model.SJ, model.SP, model.ST, within=NonNegativeReals, bounds=tca_bound)
# Auxiliary variable for the bilinear term w35_1 = tca * fcpc.
model.w35_1 = Var(model.SJ, model.SP, model.ST, within=NonNegativeReals)
# Auxiliary variable for the bilinear term w35_2 = tc * fcpc.
model.w35_2 = Var(model.SJ, model.SP, model.ST, within=NonNegativeReals)

# McCormick variables for rule 34 (linearization of the condenser balance).
model.fcpc = Var(model.SJ, model.SP, model.SK, within=NonNegativeReals, bounds=(10, 1000))  # condenser-side FCp
model.dtcj = Var(model.SJ, model.ST, within=NonNegativeReals)                               # tc[j,k]-tc[j,k+1]
model.y34 = Var(model.SJ, model.SP, model.ST, within=NonNegativeReals)                      # fcpc * dtcj

# Area lower-bound variables (linear capital cost).
model.A_hx = Var(model.SI, model.SJ, model.ST, within=NonNegativeReals)  # process-process exchanger area
model.A_cu = Var(model.SI, within=NonNegativeReals)                      # cooler area
model.A_hu = Var(model.SJ, within=NonNegativeReals)                      # heater area

# ---- Heat-pump cost variables (added) ----
# Binary: 1 if heat pump p is installed (has at least one match).
model.yp = Var(model.SP, within=Binary)
# Evaporator area lower bound of the match (i,p,k).
model.A_eva = Var(model.SI, model.SP, model.ST, within=NonNegativeReals)
# Condenser area lower bound of the match (j,p,k).
model.A_con = Var(model.SJ, model.SP, model.ST, within=NonNegativeReals)

# Precomputed constant coefficient: coef35[p] = cphp[p]*(thpcin[p]-thpca[p]).
coef35 = {p: heatpump2.cphp[p] * (heatpump2.thpcin[p] - heatpump2.thpca[p]) for p in heatpump2.SP}
# Variable bound definitions for the McCormick envelopes.
TCA_L = {j: heatpump2.tcin[j] for j in heatpump2.SJ}   # lower bound of tca
TCA_U = {j: heatpump2.tcout[j] for j in heatpump2.SJ}  # upper bound of tca
TC_L = {j: heatpump2.tcin[j] for j in heatpump2.SJ}    # lower bound of tc
TC_U = {j: heatpump2.tcout[j] for j in heatpump2.SJ}   # upper bound of tc
FCPC_L = 10.0                                          # lower bound of fcpc
FCPC_U = 1000.0                                        # upper bound of fcpc

# --------------------------
# Constraints
# --------------------------
# Overall heat balance of hot stream i.
def constrs_rule2(model, i):
    return (sum(model.q[i, j, k] for j in model.SJ for k in model.ST) +
            sum(model.qhph[i, p, k] for p in model.SP for k in model.ST) + model.qc[i]
            == heatpump2.fh[i] * (heatpump2.thin[i] - heatpump2.thout[i]))

# Overall heat balance of cold stream j.
def constrs_rule3(model, j):
    return (sum(model.q[i, j, k] for i in model.SI for k in model.ST) +
            sum(model.qhpc[j, p, k] for p in model.SP for k in model.ST) + model.qh[j]
            == heatpump2.fc[j] * (heatpump2.tcout[j] - heatpump2.tcin[j]))

# Exchanger duty is zero unless the match exists (big-M bound).
def constrs_rule4(model, i, j, k):
    return model.q[i, j, k] <= model.z[i, j, k] * min(heatpump2.ech[i], heatpump2.ecc[j])

# Cooler duty is zero unless the cooler exists.
def constrs_rule5(model, i):
    return model.qc[i] <= model.zcu[i] * heatpump2.ech[i]

# Heater duty is zero unless the heater exists.
def constrs_rule6(model, j):
    return model.qh[j] <= model.zhu[j] * heatpump2.ecc[j]

# Condenser duty is zero unless the condenser match exists.
def constrs_rule7(model, j, p, k):
    return model.qhpc[j, p, k] <= model.zhpc[j, p, k] * heatpump2.ecc[j]

# Evaporator duty is zero unless the evaporator match exists.
def constrs_rule8(model, i, p, k):
    return model.qhph[i, p, k] <= model.zhph[i, p, k] * heatpump2.ech[i]

# At most one evaporator match between hot stream i and heat pump p.
def constrs_rule9(model, i, p):
    return sum(model.zhph[i, p, k] for k in model.ST) <= 1

# At most one condenser match between cold stream j and heat pump p.
def constrs_rule10(model, j, p):
    return sum(model.zhpc[j, p, k] for k in model.ST) <= 1

# Evaporator branch limit of heat pump p.
def constrs_rule11(model, p):
    return sum(model.zhph[i, p, k] for i in model.SI for k in model.ST) <= heatpump2.nbe[p]

# Condenser branch limit of heat pump p.
def constrs_rule12(model, p):
    return sum(model.zhpc[j, p, k] for j in model.SJ for k in model.ST) <= heatpump2.nbc[p]

# Evaporator energy balance: duty = latent heat * mass flow * (1 - vapor fraction after expansion).
def constrs_rule13(model, i, p, k):
    return model.qhph[i, p, k] == (heatpump2.evap[p]) * model.mhph[i, p, k] * (1 - heatpump2.hvap[p])

# Total heat-pump fluid mass flow equals the sum over its evaporator matches.
def constrs_rule14(model, p):
    return model.mhp[p] == sum(model.mhph[i, p, k] for i in model.SI for k in model.ST)

# Condenser energy balance: duty = (latent heat + sensible desuperheat) * mass flow.
def constrs_rule15(model, j, p, k):
    return model.qhpc[j, p, k] == (
            heatpump2.cond[p] + heatpump2.cphp[p] * (heatpump2.thpcin[p] - heatpump2.thpca[p])
    ) * model.mhpc[j, p, k]

# Total heat-pump fluid mass flow equals the sum over its condenser matches.
def constrs_rule16(model, p):
    return model.mhp[p] == sum(model.mhpc[j, p, k] for j in model.SJ for k in model.ST)

# Overall energy balance of heat pump p: evaporator duties + compressor power = condenser duties.
def constrs_rule17(model, p):
    return sum(model.qhph[i, p, k] for i in model.SI for k in model.ST) + model.w[p] == \
        sum(model.qhpc[j, p, k] for j in model.SJ for k in model.ST)

# Stage heat balance of hot stream i.
def constrs_rule18(model, i, k):
    return sum(model.q[i, j, k] for j in model.SJ) + sum(model.qhph[i, p, k] for p in model.SP) == \
        heatpump2.fh[i] * (model.th[i, k] - model.th[i, k + 1])

# Stage heat balance of cold stream j.
def constrs_rule19(model, j, k):
    return sum(model.q[i, j, k] for i in model.SI) + sum(model.qhpc[j, p, k] for p in model.SP) == \
        heatpump2.fc[j] * (model.tc[j, k] - model.tc[j, k + 1])

# Supply temperature of hot stream i at the first stage.
def constrs_rule20(model, i):
    return model.th[i, 0] == heatpump2.thin[i]

# Supply temperature of cold stream j at the last stage.
def constrs_rule21(model, j):
    return model.tc[j, heatpump2.NK - 1] == heatpump2.tcin[j]

# Hot-stream temperature is monotonically non-increasing.
def constrs_rule22(model, i, k):
    return model.th[i, k] >= model.th[i, k + 1]

# Hot-stream temperature before the cooler reaches at least its target.
def constrs_rule23(model, i):
    return model.th[i, heatpump2.NK - 1] >= heatpump2.thout[i]

# Cold-stream temperature is monotonically non-decreasing.
def constrs_rule24(model, j, k):
    return model.tc[j, k] >= model.tc[j, k + 1]

# Cold-stream temperature before the heater does not exceed its target.
def constrs_rule25(model, j):
    return model.tc[j, 0] <= heatpump2.tcout[j]

# Cooler duty definition.
def constrs_rule26(model, i):
    return model.qc[i] == heatpump2.fh[i] * (model.th[i, heatpump2.NK - 1] - heatpump2.thout[i])

# Heater duty definition.
def constrs_rule27(model, j):
    return model.qh[j] == heatpump2.fc[j] * (heatpump2.tcout[j] - model.tc[j, 0])

# Exchanger approach temperature at the hot end.
def constrs_rule28(model, i, j, k):
    return model.dt[i, j, k] <= model.th[i, k] - model.tc[j, k] + (1 - model.z[i, j, k]) * heatpump2.gamma[i, j]

# Exchanger approach temperature at the cold end.
def constrs_rule29(model, i, j, k):
    return model.dt[i, j, k + 1] <= model.th[i, k + 1] - model.tc[j, k + 1] + (1 - model.z[i, j, k]) * heatpump2.gamma[i, j]

# Cooler approach temperature.
def constrs_rule30(model, i):
    return model.dtcu[i] <= model.th[i, heatpump2.NK - 1] - heatpump2.tcuout + (1 - model.zcu[i]) * heatpump2.gammacu[i]

# Heater approach temperature.
def constrs_rule31(model, j):
    return model.dthu[j] <= heatpump2.thuout - model.tc[j, 0] + (1 - model.zhu[j]) * heatpump2.gammahu[j]

# Evaporator approach temperature at the hot end.
def constrs_rule32(model, i, p, k):
    return model.dthph[i, p, k] <= model.th[i, k] - heatpump2.thpeout[p] + (1 - model.zhph[i, p, k]) * heatpump2.gammae[i, p, k]

# Evaporator approach temperature at the cold end.
def constrs_rule33(model, i, p, k):
    return model.dthph[i, p, k + 1] <= model.th[i, k + 1] - heatpump2.thpein[p] + (1 - model.zhph[i, p, k]) * \
        heatpump2.gammae[i, p, k]

# Condenser approach temperature at the hot end (desuperheat inlet).
def constrs_rule36(model, j, k, p):
    return model.dthpc[j, k, p] <= heatpump2.thpcin[p] - model.tc[j, k] + (1 - model.zhpc[j, p, k]) * heatpump2.gammajk[j, k]

# Condenser approach temperature at the intermediate (dew-point) point.
def constrs_rule37(model, j, k, p):
    return model.dthpcs[j, k, p] <= heatpump2.thpca[p] - model.tc[j, k] + (1 - model.zhpc[j, p, k]) * heatpump2.gammajk[j, k]

# Condenser approach temperature at the cold end (condensate outlet).
def constrs_rule38(model, j, k, p):
    return model.dthpc[j, k + 1, p] <= heatpump2.thpcout[p] - model.tc[j, k + 1] + (1 - model.zhpc[j, p, k]) * \
        heatpump2.gammajk[j, k]

# McCormick linearization of rule 34: qhpc = fcpc * dtcj.
def constrs_rule34_dtcj(model, j, k):
    return model.dtcj[j, k] == model.tc[j, k] - model.tc[j, k + 1]

DTL = {j: 0.0 for j in heatpump2.SJ}
DTU = {j: float(heatpump2.tcout[j] - heatpump2.tcin[j]) for j in heatpump2.SJ}
FL, FU = 10.0, 1000.0

# McCormick convex under-estimator 1.
def constrs_rule34_mcc1(model, j, p, k):
    xL = DTL[j]
    return model.y34[j, p, k] >= FL * model.dtcj[j, k] + xL * model.fcpc[j, p, k] - FL * xL

# McCormick convex under-estimator 2.
def constrs_rule34_mcc2(model, j, p, k):
    xU = DTU[j]
    return model.y34[j, p, k] >= FU * model.dtcj[j, k] + xU * model.fcpc[j, p, k] - FU * xU

# McCormick concave over-estimator 1.
def constrs_rule34_mcc3(model, j, p, k):
    xL = DTL[j]
    return model.y34[j, p, k] <= FU * model.dtcj[j, k] + xL * model.fcpc[j, p, k] - FU * xL

# McCormick concave over-estimator 2.
def constrs_rule34_mcc4(model, j, p, k):
    xU = DTU[j]
    return model.y34[j, p, k] <= FL * model.dtcj[j, k] + xU * model.fcpc[j, p, k] - FL * xU

# Link the McCormick product variable to the condenser duty.
def constrs_rule34_link(model, j, p, k):
    return model.qhpc[j, p, k] == model.y34[j, p, k]

# Bilinear term 1: relaxation of w35_1 = tca * fcpc (McCormick envelope).
def constrs_rule35_mcc1_1(model, j, p, k):
    return model.w35_1[j, p, k] >= TCA_L[j] * model.fcpc[j, p, k] + FCPC_L * model.tca[j, p, k] - TCA_L[j] * FCPC_L

def constrs_rule35_mcc1_2(model, j, p, k):
    return model.w35_1[j, p, k] >= TCA_U[j] * model.fcpc[j, p, k] + FCPC_U * model.tca[j, p, k] - TCA_U[j] * FCPC_U

def constrs_rule35_mcc1_3(model, j, p, k):
    return model.w35_1[j, p, k] <= TCA_U[j] * model.fcpc[j, p, k] + FCPC_L * model.tca[j, p, k] - TCA_U[j] * FCPC_L

def constrs_rule35_mcc1_4(model, j, p, k):
    return model.w35_1[j, p, k] <= TCA_L[j] * model.fcpc[j, p, k] + FCPC_U * model.tca[j, p, k] - TCA_L[j] * FCPC_U

# Bilinear term 2: relaxation of w35_2 = tc * fcpc (McCormick envelope).
def constrs_rule35_mcc2_1(model, j, p, k):
    return model.w35_2[j, p, k] >= TC_L[j] * model.fcpc[j, p, k] + FCPC_L * model.tc[j, k] - TC_L[j] * FCPC_L

def constrs_rule35_mcc2_2(model, j, p, k):
    return model.w35_2[j, p, k] >= TC_U[j] * model.fcpc[j, p, k] + FCPC_U * model.tc[j, k] - TC_U[j] * FCPC_U

def constrs_rule35_mcc2_3(model, j, p, k):
    return model.w35_2[j, p, k] <= TC_U[j] * model.fcpc[j, p, k] + FCPC_L * model.tc[j, k] - TC_U[j] * FCPC_L

def constrs_rule35_mcc2_4(model, j, p, k):
    return model.w35_2[j, p, k] <= TC_L[j] * model.fcpc[j, p, k] + FCPC_U * model.tc[j, k] - TC_L[j] * FCPC_U

# Linearization of the original constraint (35): w35_1 = w35_2 - coef35 * mhpc.
def constrs_rule35_link(model, j, p, k):
    return model.w35_1[j, p, k] == model.w35_2[j, p, k] - coef35[p] * model.mhpc[j, p, k]

# ---- Heat-pump installation binary links (added) ----
# yp[p] = 1 if heat pump p has any evaporator match.
def constrs_rule_yp_eva(model, p):
    return sum(model.zhph[i, p, k] for i in model.SI for k in model.ST) <= heatpump2.nbe[p] * model.yp[p]

# yp[p] = 1 if heat pump p has any condenser match.
def constrs_rule_yp_con(model, p):
    return sum(model.zhpc[j, p, k] for j in model.SJ for k in model.ST) <= heatpump2.nbc[p] * model.yp[p]

# --------------------------
# Area lower-bound data (reference LMTDs)
# --------------------------
# Combined thermal resistance of exchanger (i,j).
Uij = {(i, j): (1.0 / heatpump2.hh[i] + 1.0 / heatpump2.hc[j]) for i in heatpump2.SI for j in heatpump2.SJ}
# Combined thermal resistance of the cooler of hot stream i.
Ucu = {i: (1.0 / heatpump2.hh[i] + 1.0 / heatpump2.hcu) for i in heatpump2.SI}
# Combined thermal resistance of the heater of cold stream j.
Uhu = {j: (1.0 / heatpump2.hc[j] + 1.0 / heatpump2.hhu) for j in heatpump2.SJ}

# ---- Heat-pump exchanger coefficients (added) ----
# Overall heat-transfer coefficient of the evaporator (hot stream i vs working fluid).
U_eva = {i: 1.0 / (1.0 / heatpump2.hh[i] + 1.0 / heatpump2.heva) for i in heatpump2.SI}
# Overall heat-transfer coefficient of the condenser (cold stream j vs working fluid).
U_con = {j: 1.0 / (1.0 / heatpump2.hc[j] + 1.0 / heatpump2.hcon) for j in heatpump2.SJ}

# Minimum approach temperature for the reference LMTD values.
EMAT = 20
# Small constant to avoid division by zero in the logarithm.
eps = 1e-6

# Safe log-mean temperature difference (falls back to the arithmetic mean when d1 == d2).
def _lmtd(d1, d2):
    if abs(d1 - d2) < 1e-9:
        return (d1 + d2) / 2.0
    return (d1 - d2) / np.log(d1 / d2 + eps)

# Reference LMTD for the process-process exchanger (i,j).
DTref_hx = {}
for i in range(len(heatpump2.thin)):
    for j in range(len(heatpump2.tcin)):
        deltaT1 = heatpump2.thin[i] - heatpump2.tcout[j]
        deltaT2 = heatpump2.thout[i] - heatpump2.tcin[j]
        deltaT1 = max(deltaT1, EMAT)
        deltaT2 = max(deltaT2, EMAT)
        DTref_hx[(i, j)] = _lmtd(deltaT1, deltaT2)

# Reference LMTD for the cooler of hot stream i.
DTref_cu = {}
for i in range(len(heatpump2.thin)):
    deltaT1 = heatpump2.thin[i] - heatpump2.tcuout
    deltaT2 = heatpump2.thout[i] - heatpump2.tcuin
    deltaT1 = max(deltaT1, EMAT)
    deltaT2 = max(deltaT2, EMAT)
    DTref_cu[i] = _lmtd(deltaT1, deltaT2)

# Reference temperature difference for the heater of cold stream j (isothermal hot utility).
DTref_hu = {}
for j in range(len(heatpump2.tcin)):
    deltaT = heatpump2.thuin - heatpump2.tcout[j]
    DTref_hu[j] = max(deltaT, EMAT)

# ---- Process exchanger / cooler / heater area lower-bound constraints (added) ----
# A_hx[i,j,k] >= q[i,j,k] / (Uij[i,j] * DTref_hx[i,j])
def constrs_rule_Ahx(model, i, j, k):
    return model.A_hx[i, j, k] >= model.q[i, j, k] / (Uij[(i, j)] * DTref_hx[(i, j)])

# A_cu[i] >= qc[i] / (Ucu[i] * DTref_cu[i])
def constrs_rule_Acu(model, i):
    return model.A_cu[i] >= model.qc[i] / (Ucu[i] * DTref_cu[i])

# A_hu[j] >= qh[j] / (Uhu[j] * DTref_hu[j])
def constrs_rule_Ahu(model, j):
    return model.A_hu[j] >= model.qh[j] / (Uhu[j] * DTref_hu[j])

# ---- Reference LMTDs for the heat-pump exchangers (added) ----
# Evaporator: hot stream i vs the evaporator working fluid.
DTref_eva = {}
for i in range(len(heatpump2.thin)):
    for p in range(len(heatpump2.thpein)):
        d1 = heatpump2.thin[i] - heatpump2.thpeout[p]
        d2 = heatpump2.thout[i] - heatpump2.thpein[p]
        d1 = max(d1, EMAT)
        d2 = max(d2, EMAT)
        DTref_eva[(i, p)] = _lmtd(d1, d2)

# Condenser: cold stream j vs the condensing working fluid.
DTref_con = {}
for j in range(len(heatpump2.tcin)):
    for p in range(len(heatpump2.thpcin)):
        d1 = heatpump2.thpcin[p] - heatpump2.tcout[j]
        d2 = heatpump2.thpcout[p] - heatpump2.tcin[j]
        d1 = max(d1, EMAT)
        d2 = max(d2, EMAT)
        DTref_con[(j, p)] = _lmtd(d1, d2)

# ---- Heat-pump area lower-bound constraints (added): A >= q / (U * DTref) ----
def constrs_rule_Aeva(model, i, p, k):
    return model.A_eva[i, p, k] >= model.qhph[i, p, k] / (U_eva[i] * DTref_eva[(i, p)])

def constrs_rule_Acon(model, j, p, k):
    return model.A_con[j, p, k] >= model.qhpc[j, p, k] / (U_con[j] * DTref_con[(j, p)])

# --------------------------
# Attach constraints
# --------------------------
model.constrs2 = Constraint(model.SI, rule=constrs_rule2)
model.constrs3 = Constraint(model.SJ, rule=constrs_rule3)
model.constrs4 = Constraint(model.SI, model.SJ, model.ST, rule=constrs_rule4)
model.constrs5 = Constraint(model.SI, rule=constrs_rule5)
model.constrs6 = Constraint(model.SJ, rule=constrs_rule6)
model.constrs7 = Constraint(model.SJ, model.SP, model.ST, rule=constrs_rule7)
model.constrs8 = Constraint(model.SI, model.SP, model.ST, rule=constrs_rule8)
model.constrs9 = Constraint(model.SI, model.SP, rule=constrs_rule9)
model.constrs10 = Constraint(model.SJ, model.SP, rule=constrs_rule10)
model.constrs11 = Constraint(model.SP, rule=constrs_rule11)
model.constrs12 = Constraint(model.SP, rule=constrs_rule12)
model.constrs13 = Constraint(model.SI, model.SP, model.ST, rule=constrs_rule13)
model.constrs14 = Constraint(model.SP, rule=constrs_rule14)
model.constrs15 = Constraint(model.SJ, model.SP, model.ST, rule=constrs_rule15)
model.constrs16 = Constraint(model.SP, rule=constrs_rule16)
model.constrs17 = Constraint(model.SP, rule=constrs_rule17)
model.constrs18 = Constraint(model.SI, model.ST, rule=constrs_rule18)
model.constrs19 = Constraint(model.SJ, model.ST, rule=constrs_rule19)
model.constrs20 = Constraint(model.SI, rule=constrs_rule20)
model.constrs21 = Constraint(model.SJ, rule=constrs_rule21)
model.constrs22 = Constraint(model.SI, model.ST, rule=constrs_rule22)
model.constrs23 = Constraint(model.SI, rule=constrs_rule23)
model.constrs24 = Constraint(model.SJ, model.ST, rule=constrs_rule24)
model.constrs25 = Constraint(model.SJ, rule=constrs_rule25)
model.constrs26 = Constraint(model.SI, rule=constrs_rule26)
model.constrs27 = Constraint(model.SJ, rule=constrs_rule27)
model.constrs28 = Constraint(model.SIJ, model.ST, rule=constrs_rule28)
model.constrs29 = Constraint(model.SIJ, model.ST, rule=constrs_rule29)
model.constrs30 = Constraint(model.SI, rule=constrs_rule30)
model.constrs31 = Constraint(model.SJ, rule=constrs_rule31)
model.constrs32 = Constraint(model.SI, model.SP, model.ST, rule=constrs_rule32)
model.constrs33 = Constraint(model.SI, model.SP, model.ST, rule=constrs_rule33)
model.constrs36 = Constraint(model.SJ, model.ST, model.SP, rule=constrs_rule36)
model.constrs37 = Constraint(model.SJ, model.ST, model.SP, rule=constrs_rule37)
model.constrs38 = Constraint(model.SJ, model.ST, model.SP, rule=constrs_rule38)

model.constrs34_dtcj = Constraint(model.SJ, model.ST, rule=constrs_rule34_dtcj)
model.constrs34_mcc1 = Constraint(model.SJ, model.SP, model.ST, rule=constrs_rule34_mcc1)
model.constrs34_mcc2 = Constraint(model.SJ, model.SP, model.ST, rule=constrs_rule34_mcc2)
model.constrs34_mcc3 = Constraint(model.SJ, model.SP, model.ST, rule=constrs_rule34_mcc3)
model.constrs34_mcc4 = Constraint(model.SJ, model.SP, model.ST, rule=constrs_rule34_mcc4)
model.constrs34_link = Constraint(model.SJ, model.SP, model.ST, rule=constrs_rule34_link)

model.constrs35_mcc1_1 = Constraint(model.SJ, model.SP, model.ST, rule=constrs_rule35_mcc1_1)
model.constrs35_mcc1_2 = Constraint(model.SJ, model.SP, model.ST, rule=constrs_rule35_mcc1_2)
model.constrs35_mcc1_3 = Constraint(model.SJ, model.SP, model.ST, rule=constrs_rule35_mcc1_3)
model.constrs35_mcc1_4 = Constraint(model.SJ, model.SP, model.ST, rule=constrs_rule35_mcc1_4)
model.constrs35_mcc2_1 = Constraint(model.SJ, model.SP, model.ST, rule=constrs_rule35_mcc2_1)
model.constrs35_mcc2_2 = Constraint(model.SJ, model.SP, model.ST, rule=constrs_rule35_mcc2_2)
model.constrs35_mcc2_3 = Constraint(model.SJ, model.SP, model.ST, rule=constrs_rule35_mcc2_3)
model.constrs35_mcc2_4 = Constraint(model.SJ, model.SP, model.ST, rule=constrs_rule35_mcc2_4)
model.constrs35_link = Constraint(model.SJ, model.SP, model.ST, rule=constrs_rule35_link)

# ---- Attach process exchanger / cooler / heater area lower bounds (added) ----
model.constrs_Ahx = Constraint(model.SI, model.SJ, model.ST, rule=constrs_rule_Ahx)
model.constrs_Acu = Constraint(model.SI, rule=constrs_rule_Acu)
model.constrs_Ahu = Constraint(model.SJ, rule=constrs_rule_Ahu)

# ---- Attach heat-pump cost constraints (added) ----
model.constrs_yp_eva = Constraint(model.SP, rule=constrs_rule_yp_eva)
model.constrs_yp_con = Constraint(model.SP, rule=constrs_rule_yp_con)
model.constrs_Aeva = Constraint(model.SI, model.SP, model.ST, rule=constrs_rule_Aeva)
model.constrs_Acon = Constraint(model.SJ, model.SP, model.ST, rule=constrs_rule_Acon)

# --------------------------
# Objective
# --------------------------
C_HX = heatpump2.acoeff
C_CU = heatpump2.cucoeff
C_HU = heatpump2.hucoeff

# Annual operating hours for the annualized electricity cost.
HOURS_PER_YEAR = 8000.0

def obj_rule(model):
    # Utility cost: hot-utility (heaters) + cold-utility (coolers).
    utility = (
        sum(model.qh[j] * heatpump2.hucost for j in model.SJ)
        + sum(model.qc[i] * heatpump2.cucost for i in model.SI)
    )

    # Annual electricity cost of the heat-pump compressors (kW -> MW -> $/yr).
    electricity = (heatpump2.ELECTRICITY_COST / 1000.0 * HOURS_PER_YEAR) * \
        sum(model.w[p] for p in model.SP)

    # Fixed capital cost of the installed heat pumps.
    hp_fixed = heatpump2.HP_FIXED_COST * sum(model.yp[p] for p in model.SP)

    # Capital area cost: exchangers + coolers + heaters + evaporators + condensers.
    cap_area = (
        C_HX * sum(model.A_hx[i, j, k] for i in model.SI for j in model.SJ for k in model.ST)
        + C_CU * sum(model.A_cu[i] for i in model.SI)
        + C_HU * sum(model.A_hu[j] for j in model.SJ)
        + heatpump2.HP_EVAP_COEFF * sum(model.A_eva[i, p, k] for i in model.SI for p in model.SP for k in model.ST)
        + heatpump2.HP_COND_COEFF * sum(model.A_con[j, p, k] for j in model.SJ for p in model.SP for k in model.ST)
    )

    # Fixed cost of all installed units, including the heat-pump match connections.
    fixed_units = heatpump2.unitc * (
        sum(model.z[i, j, k] for i in model.SI for j in model.SJ for k in model.ST)
        + sum(model.zcu[i] for i in model.SI)
        + sum(model.zhu[j] for j in model.SJ)
    ) + heatpump2.HP_UNIT_COST * (
        sum(model.zhph[i, p, k] for i in model.SI for p in model.SP for k in model.ST)
        + sum(model.zhpc[j, p, k] for j in model.SJ for p in model.SP for k in model.ST)
    )

    # Total annualized cost to be minimized.
    return utility + electricity + hp_fixed + cap_area + fixed_units

model.obj = Objective(rule=obj_rule, sense=minimize)

# --------------------------
# Solve
# --------------------------
solver = SolverFactory("glpk")
res = solver.solve(model, tee=True)

print("\n" + "=" * 60)
print("MILP-LB RESULT")
print("=" * 60)
print("Status:", res.solver.status)
print("Termination:", res.solver.termination_condition)

if (res.solver.termination_condition == TerminationCondition.optimal or
        res.solver.termination_condition == TerminationCondition.feasible):
    print("LB objective =", value(model.obj))

    print("\nActive z(i,j,k):")
    for i in model.SI:
        for j in model.SJ:
            for k in model.ST:
                if value(model.z[i, j, k]) > 0.5:
                    print(f"z[{i},{j},{k}] = 1")

    print("\nActive zhu, zcu:")
    for j in model.SJ:
        if value(model.zhu[j]) > 0.5:
            print(f"zhu[{j}] = 1")
    for i in model.SI:
        if value(model.zcu[i]) > 0.5:
            print(f"zcu[{i}] = 1")

    print("\nActive zhph, zhpc:")
    for i in model.SI:
        for p in model.SP:
            for k in model.ST:
                if value(model.zhph[i, p, k]) > 0.5:
                    print(f"zhph[{i},{p},{k}] = 1")
    for j in model.SJ:
        for p in model.SP:
            for k in model.ST:
                if value(model.zhpc[j, p, k]) > 0.5:
                    print(f"zhpc[{j},{p},{k}] = 1")
else:
    print("No feasible LB solution found.")

print(f"\nElapsed: {time.time() - start_time:.2f} s")
