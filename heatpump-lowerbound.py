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


def _SIJ_init(model):
    return ((i, j) for i in model.SI for j in model.SJ
            if (heatpump2.thin[i] - heatpump2.tcin[j] >= heatpump2.EMAT))


model.SIJ = Set(dimen=2, initialize=_SIJ_init)


# --------------------------
# Variables
# tca变量定义
def tca_bound(model, j, p, k):
    # tca上下界与冷物流温度范围一致
    return (heatpump2.tcin[j], heatpump2.tcout[j])


# --------------------------
def q_bound(model, i, j, k):
    return (0, min(heatpump2.ech[i], heatpump2.ecc[j]))


model.q = Var(model.SI, model.SJ, model.SK, within=NonNegativeReals, bounds=q_bound)


def qhph_bound(model, i, p, k):
    return (0, heatpump2.ech[i])


model.qhph = Var(model.SI, model.SP, model.SK, within=NonNegativeReals, bounds=qhph_bound)


def qhpc_bound(model, j, p, k):
    return (0, heatpump2.ecc[j])


model.qhpc = Var(model.SJ, model.SP, model.SK, within=NonNegativeReals, bounds=qhpc_bound)

model.mhph = Var(model.SI, model.SP, model.SK, within=NonNegativeReals, bounds=(0, 10000))
model.mhpc = Var(model.SJ, model.SP, model.SK, within=NonNegativeReals, bounds=(0, 10000))
model.mhp = Var(model.SP, within=NonNegativeReals, bounds=(0, 10000))


def qc_bound(model, i):
    return (0, heatpump2.ech[i])


def qh_bound(model, j):
    return (0, heatpump2.ecc[j])


model.qc = Var(model.SI, within=NonNegativeReals, bounds=qc_bound)
model.qh = Var(model.SJ, within=NonNegativeReals, bounds=qh_bound)


def th_bound(model, i, k):
    return (heatpump2.thout[i], heatpump2.thin[i])


def tc_bound(model, j, k):
    return (heatpump2.tcin[j], heatpump2.tcout[j])


model.th = Var(model.SI, model.SK, within=NonNegativeReals, bounds=th_bound)
model.tc = Var(model.SJ, model.SK, within=NonNegativeReals, bounds=tc_bound)


def w_bound(model, p):
    return (0, heatpump2.wmax[p])


model.w = Var(model.SP, within=NonNegativeReals, bounds=w_bound)


def dt_bound(model, i, j, k):
    return (heatpump2.EMAT, heatpump2.gamma[i, j])


model.dt = Var(model.SI, model.SJ, model.SK, within=NonNegativeReals, bounds=dt_bound)


def dtcu_bound(model, i):
    return (heatpump2.EMAT, heatpump2.gammacu[i])


def dthu_bound(model, j):
    return (heatpump2.EMAT, heatpump2.gammahu[j])


def dthph_bound(model, i, p, k):
    return (heatpump2.EMAT, heatpump2.gammae[i, p, k])


def dthpc_bound(model, j, k, p):
    return (heatpump2.EMAT, heatpump2.gammajk[j, k])


model.dtcu = Var(model.SI, within=NonNegativeReals, bounds=dtcu_bound)
model.dthu = Var(model.SJ, within=NonNegativeReals, bounds=dthu_bound)
model.dthph = Var(model.SI, model.SP, model.SK, within=NonNegativeReals, bounds=dthph_bound)
model.dthpc = Var(model.SJ, model.SK, model.SP, within=NonNegativeReals, bounds=dthpc_bound)
model.dthpcs = Var(model.SJ, model.SK, model.SP, within=NonNegativeReals, bounds=dthpc_bound)

# binaries
model.z = Var(model.SI, model.SJ, model.SK, within=Binary)
model.zhph = Var(model.SI, model.SP, model.SK, within=Binary)
model.zhpc = Var(model.SJ, model.SP, model.SK, within=Binary)
model.zhu = Var(model.SJ, within=Binary)
model.zcu = Var(model.SI, within=Binary)

model.tca = Var(model.SJ, model.SP, model.ST, within=NonNegativeReals, bounds=tca_bound)
# 双线性项辅助变量：w35_1 = tca * fcpc
model.w35_1 = Var(model.SJ, model.SP, model.ST, within=NonNegativeReals)
# 双线性项辅助变量：w35_2 = tc * fcpc
model.w35_2 = Var(model.SJ, model.SP, model.ST, within=NonNegativeReals)

# McCormick vars for rule34
model.fcpc = Var(model.SJ, model.SP, model.SK, within=NonNegativeReals, bounds=(10, 1000))
model.dtcj = Var(model.SJ, model.ST, within=NonNegativeReals)
model.y34 = Var(model.SJ, model.SP, model.ST, within=NonNegativeReals)

# area lower-bound vars (linear cost)
model.A_hx = Var(model.SI, model.SJ, model.ST, within=NonNegativeReals)
model.A_cu = Var(model.SI, within=NonNegativeReals)
model.A_hu = Var(model.SJ, within=NonNegativeReals)


# 预计算常数系数
coef35 = {p: heatpump2.cphp[p] * (heatpump2.thpcin[p] - heatpump2.thpca[p]) for p in heatpump2.SP}
# 变量边界定义
TCA_L = {j: heatpump2.tcin[j] for j in heatpump2.SJ}  # tca下界
TCA_U = {j: heatpump2.tcout[j] for j in heatpump2.SJ}  # tca上界
TC_L = {j: heatpump2.tcin[j] for j in heatpump2.SJ}  # tc下界
TC_U = {j: heatpump2.tcout[j] for j in heatpump2.SJ}  # tc上界
FCPC_L = 10.0  # fcpc下界（与原有变量定义一致）
FCPC_U = 1000.0  # fcpc上界（与原有变量定义一致）


# --------------------------
# Constraints
# --------------------------
def constrs_rule2(model, i):
    return (sum(model.q[i, j, k] for j in model.SJ for k in model.ST) +
            sum(model.qhph[i, p, k] for p in model.SP for k in model.ST) + model.qc[i]
            == heatpump2.fh[i] * (heatpump2.thin[i] - heatpump2.thout[i]))


def constrs_rule3(model, j):
    return (sum(model.q[i, j, k] for i in model.SI for k in model.ST) +
            sum(model.qhpc[j, p, k] for p in model.SP for k in model.ST) + model.qh[j]
            == heatpump2.fc[j] * (heatpump2.tcout[j] - heatpump2.tcin[j]))


def constrs_rule4(model, i, j, k):
    return model.q[i, j, k] <= model.z[i, j, k] * min(heatpump2.ech[i], heatpump2.ecc[j])


def constrs_rule5(model, i):
    return model.qc[i] <= model.zcu[i] * heatpump2.ech[i]


def constrs_rule6(model, j):
    return model.qh[j] <= model.zhu[j] * heatpump2.ecc[j]


def constrs_rule7(model, j, p, k):
    return model.qhpc[j, p, k] <= model.zhpc[j, p, k] * heatpump2.ecc[j]


def constrs_rule8(model, i, p, k):
    return model.qhph[i, p, k] <= model.zhph[i, p, k] * heatpump2.ech[i]


def constrs_rule9(model, i, p):
    return sum(model.zhph[i, p, k] for k in model.ST) <= 1


def constrs_rule10(model, j, p):
    return sum(model.zhpc[j, p, k] for k in model.ST) <= 1


def constrs_rule11(model, p):
    return sum(model.zhph[i, p, k] for i in model.SI for k in model.ST) <= heatpump2.nbe[p]


def constrs_rule12(model, p):
    return sum(model.zhpc[j, p, k] for j in model.SJ for k in model.ST) <= heatpump2.nbc[p]


def constrs_rule13(model, i, p, k):
    return model.qhph[i, p, k] == (heatpump2.evap[p]) * model.mhph[i, p, k] * (1 - heatpump2.hvap[p])


def constrs_rule14(model, p):
    return model.mhp[p] == sum(model.mhph[i, p, k] for i in model.SI for k in model.ST)


def constrs_rule15(model, j, p, k):
    return model.qhpc[j, p, k] == (
            heatpump2.cond[p] + heatpump2.cphp[p] * (heatpump2.thpcin[p] - heatpump2.thpca[p])
    ) * model.mhpc[j, p, k]


def constrs_rule16(model, p):
    return model.mhp[p] == sum(model.mhpc[j, p, k] for j in model.SJ for k in model.ST)


def constrs_rule17(model, p):
    return sum(model.qhph[i, p, k] for i in model.SI for k in model.ST) + model.w[p] == \
        sum(model.qhpc[j, p, k] for j in model.SJ for k in model.ST)


def constrs_rule18(model, i, k):
    return sum(model.q[i, j, k] for j in model.SJ) + sum(model.qhph[i, p, k] for p in model.SP) == \
        heatpump2.fh[i] * (model.th[i, k] - model.th[i, k + 1])


def constrs_rule19(model, j, k):
    return sum(model.q[i, j, k] for i in model.SI) + sum(model.qhpc[j, p, k] for p in model.SP) == \
        heatpump2.fc[j] * (model.tc[j, k] - model.tc[j, k + 1])


def constrs_rule20(model, i):
    return model.th[i, 0] == heatpump2.thin[i]


def constrs_rule21(model, j):
    return model.tc[j, heatpump2.NK - 1] == heatpump2.tcin[j]


def constrs_rule22(model, i, k):
    return model.th[i, k] >= model.th[i, k + 1]


def constrs_rule23(model, i):
    return model.th[i, heatpump2.NK - 1] >= heatpump2.thout[i]


def constrs_rule24(model, j, k):
    return model.tc[j, k] >= model.tc[j, k + 1]


def constrs_rule25(model, j):
    return model.tc[j, 0] <= heatpump2.tcout[j]


def constrs_rule26(model, i):
    return model.qc[i] == heatpump2.fh[i] * (model.th[i, heatpump2.NK - 1] - heatpump2.thout[i])


def constrs_rule27(model, j):
    return model.qh[j] == heatpump2.fc[j] * (heatpump2.tcout[j] - model.tc[j, 0])


def constrs_rule28(model, i, j, k):
    return model.dt[i, j, k] <= model.th[i, k] - model.tc[j, k] + (1 - model.z[i, j, k]) * heatpump2.gamma[i, j]


def constrs_rule29(model, i, j, k):
    return model.dt[i, j, k + 1] <= model.th[i, k + 1] - model.tc[j, k + 1] + (1 - model.z[i, j, k]) * heatpump2.gamma[
        i, j]


def constrs_rule30(model, i):
    return model.dtcu[i] <= model.th[i, heatpump2.NK - 1] - heatpump2.tcuout + (1 - model.zcu[i]) * heatpump2.gammacu[i]


def constrs_rule31(model, j):
    return model.dthu[j] <= heatpump2.thuout - model.tc[j, 0] + (1 - model.zhu[j]) * heatpump2.gammahu[j]


def constrs_rule32(model, i, p, k):
    return model.dthph[i, p, k] <= model.th[i, k] - heatpump2.thpeout[p] + (1 - model.zhph[i, p, k]) * heatpump2.gammae[
        i, p, k]


def constrs_rule33(model, i, p, k):
    return model.dthph[i, p, k + 1] <= model.th[i, k + 1] - heatpump2.thpein[p] + (1 - model.zhph[i, p, k]) * \
        heatpump2.gammae[i, p, k]


def constrs_rule36(model, j,k, p):
    return model.dthpc[j, k,p] <= heatpump2.thpcin[p] - model.tc[j, k] + (1 - model.zhpc[j, p, k]) * heatpump2.gammajk[
        j, k]


def constrs_rule37(model, j, k, p):
    return model.dthpcs[j, k, p] <= heatpump2.thpca[p] - model.tc[j, k] + (1 - model.zhpc[j, p, k]) * heatpump2.gammajk[
        j, k]


def constrs_rule38(model, j, k, p):
    return model.dthpc[j, k + 1,p] <= heatpump2.thpcout[p] - model.tc[j, k + 1] + (1 - model.zhpc[j, p, k]) * \
        heatpump2.gammajk[j, k]


# McCormick (rule34 replacement)
def constrs_rule34_dtcj(model, j, k):
    return model.dtcj[j, k] == model.tc[j, k] - model.tc[j, k + 1]


DTL = {j: 0.0 for j in heatpump2.SJ}
DTU = {j: float(heatpump2.tcout[j] - heatpump2.tcin[j]) for j in heatpump2.SJ}
FL, FU = 10.0, 1000.0


def constrs_rule34_mcc1(model, j, p, k):
    xL = DTL[j]
    return model.y34[j, p, k] >= FL * model.dtcj[j, k] + xL * model.fcpc[j, p, k] - FL * xL


def constrs_rule34_mcc2(model, j, p, k):
    xU = DTU[j]
    return model.y34[j, p, k] >= FU * model.dtcj[j, k] + xU * model.fcpc[j, p, k] - FU * xU


def constrs_rule34_mcc3(model, j, p, k):
    xL = DTL[j]
    return model.y34[j, p, k] <= FU * model.dtcj[j, k] + xL * model.fcpc[j, p, k] - FU * xL


def constrs_rule34_mcc4(model, j, p, k):
    xU = DTU[j]
    return model.y34[j, p, k] <= FL * model.dtcj[j, k] + xU * model.fcpc[j, p, k] - FL * xU


def constrs_rule34_link(model, j, p, k):
    return model.qhpc[j, p, k] == model.y34[j, p, k]


# 双线性项1：w35_1 = tca * fcpc 松弛约束
# ======================
def constrs_rule35_mcc1_1(model, j, p, k):
    # w >= xL*y + yL*x - xL*yL
    return model.w35_1[j, p, k] >= TCA_L[j] * model.fcpc[j, p, k] + FCPC_L * model.tca[j, p, k] - TCA_L[j] * FCPC_L


def constrs_rule35_mcc1_2(model, j, p, k):
    # w >= xU*y + yU*x - xU*yU
    return model.w35_1[j, p, k] >= TCA_U[j] * model.fcpc[j, p, k] + FCPC_U * model.tca[j, p, k] - TCA_U[j] * FCPC_U


def constrs_rule35_mcc1_3(model, j, p, k):
    # w <= xU*y + yL*x - xU*yL
    return model.w35_1[j, p, k] <= TCA_U[j] * model.fcpc[j, p, k] + FCPC_L * model.tca[j, p, k] - TCA_U[j] * FCPC_L


def constrs_rule35_mcc1_4(model, j, p, k):
    # w <= xL*y + yU*x - xL*yU
    return model.w35_1[j, p, k] <= TCA_L[j] * model.fcpc[j, p, k] + FCPC_U * model.tca[j, p, k] - TCA_L[j] * FCPC_U


# ======================
# 双线性项2：w35_2 = tc * fcpc 松弛约束
# ======================
def constrs_rule35_mcc2_1(model, j, p, k):
    # w >= xL*y + yL*x - xL*yL
    return model.w35_2[j, p, k] >= TC_L[j] * model.fcpc[j, p, k] + FCPC_L * model.tc[j, k] - TC_L[j] * FCPC_L


def constrs_rule35_mcc2_2(model, j, p, k):
    # w >= xU*y + yU*x - xU*yU
    return model.w35_2[j, p, k] >= TC_U[j] * model.fcpc[j, p, k] + FCPC_U * model.tc[j, k] - TC_U[j] * FCPC_U


def constrs_rule35_mcc2_3(model, j, p, k):
    # w <= xU*y + yL*x - xU*yL
    return model.w35_2[j, p, k] <= TC_U[j] * model.fcpc[j, p, k] + FCPC_L * model.tc[j, k] - TC_U[j] * FCPC_L


def constrs_rule35_mcc2_4(model, j, p, k):
    # w <= xL*y + yU*x - xL*yU
    return model.w35_2[j, p, k] <= TC_L[j] * model.fcpc[j, p, k] + FCPC_U * model.tc[j, k] - TC_L[j] * FCPC_U

# ======================
# 原约束线性化替换
# ======================
def constrs_rule35_link(model, j, p, k):
    # 原约束：tca*fcpc = tc*fcpc - C*mhpc → w35_1 = w35_2 - C*mhpc
    return model.w35_1[j, p, k] == model.w35_2[j, p, k] - coef35[p] * model.mhpc[j, p, k]


# Area LB constraints
Uij = {(i, j): (1.0 / heatpump2.hh[i] + 1.0 / heatpump2.hc[j]) for i in heatpump2.SI for j in heatpump2.SJ}
Ucu = {i: (1.0 / heatpump2.hh[i] + 1.0 / heatpump2.hcu) for i in heatpump2.SI}
Uhu = {j: (1.0 / heatpump2.hc[j] + 1.0 / heatpump2.hhu) for j in heatpump2.SJ}

EMAT = 20
eps = 1e-6

# 普通换热器DTref_hx
DTref_hx = {}
for i in range(len(heatpump2.thin)):
    for j in range(len(heatpump2.tcin)):
        deltaT1 = heatpump2.thin[i] - heatpump2.tcout[j]
        deltaT2 = heatpump2.thout[i] - heatpump2.tcin[j]

        deltaT1 = max(deltaT1, EMAT)
        deltaT2 = max(deltaT2, EMAT)
        DTref_hx[(i, j)] = (deltaT1 - deltaT2) / np.log(deltaT1 / deltaT2 + eps)

# 冷却器DTref_cu
DTref_cu = {}
for i in range(len(heatpump2.thin)):
    deltaT1 = heatpump2.thin[i] - heatpump2.tcuout
    deltaT2 = heatpump2.thout[i] - heatpump2.tcuin
    deltaT1 = max(deltaT1, EMAT)
    deltaT2 = max(deltaT2, EMAT)
    DTref_cu[i] = (deltaT1 - deltaT2) / np.log(deltaT1 / deltaT2 + eps)

# 加热器DTref_hu（恒温介质简化计算）
DTref_hu = {}
for j in range(len(heatpump2.tcin)):
    deltaT = heatpump2.thuin - heatpump2.tcout[j]
    DTref_hu[j] = max(deltaT, EMAT)





# Attach constraints
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




# --------------------------
# Objective
# --------------------------
C_HX = heatpump2.acoeff
C_CU = heatpump2.cucoeff
C_HU = heatpump2.hucoeff

def obj_rule(model):
    utility = (
        sum(model.qh[j] * heatpump2.hucost for j in model.SJ)
        + sum(model.qc[i] * heatpump2.cucost for i in model.SI)
    )

    cap_area = (
        C_HX * sum(model.A_hx[i, j, k] for i in model.SI for j in model.SJ for k in model.ST)
        + C_CU * sum(model.A_cu[i] for i in model.SI)
        + C_HU * sum(model.A_hu[j] for j in model.SJ)
    )

    fixed_units = heatpump2.unitc * (
        sum(model.z[i, j, k] for i in model.SI for j in model.SJ for k in model.ST)
        + sum(model.zcu[i] for i in model.SI)
        + sum(model.zhu[j] for j in model.SJ)
    )

    return utility + cap_area + fixed_units


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