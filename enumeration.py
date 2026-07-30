
import time
import numpy as np
import heatpump2
from pyomo.environ import *

start_time = time.time()
model = ConcreteModel()

# ============================================================
# 1) Sets
# ============================================================
model.SI = Set(initialize=[i for i in heatpump2.SI])  # hot streams
model.SJ = Set(initialize=[j for j in heatpump2.SJ])  # cold streams
model.SK = Set(initialize=[k for k in heatpump2.SK])  # nodes 0..NK-1
model.ST = Set(initialize=[k for k in heatpump2.ST])  # stages 0..NK-2
model.SP = Set(initialize=[p for p in heatpump2.SP])  # heat pumps


def _SIJ_init(m):
    return (
        (i, j)
        for i in m.SI for j in m.SJ
        if (heatpump2.thin[i] - heatpump2.tcin[j] >= heatpump2.EMAT)
    )


model.SIJ = Set(dimen=2, initialize=_SIJ_init)

# ============================================================
# 2) Variables
#    注：binary 全部放在 ST（阶段）上，避免末级虚位
# ============================================================
def q_bound(m, i, j, k):
    return (0, float(min(heatpump2.ech[i], heatpump2.ecc[j])))


model.q = Var(model.SI, model.SJ, model.ST, within=NonNegativeReals, bounds=q_bound)


def qhph_bound(m, i, p, k):
    return (0, float(heatpump2.ech[i]))


model.qhph = Var(model.SI, model.SP, model.ST, within=NonNegativeReals, bounds=qhph_bound)


def qhpc_bound(m, j, p, k):
    return (0, float(heatpump2.ecc[j]))


model.qhpc = Var(model.SJ, model.SP, model.ST, within=NonNegativeReals, bounds=qhpc_bound)

model.mhph = Var(model.SI, model.SP, model.ST, within=NonNegativeReals, bounds=(0, 10000))
model.mhpc = Var(model.SJ, model.SP, model.ST, within=NonNegativeReals, bounds=(0, 10000))
model.mhp = Var(model.SP, within=NonNegativeReals, bounds=(0, 10000))


def qc_bound(m, i):
    return (0, float(heatpump2.ech[i]))


def qh_bound(m, j):
    return (0, float(heatpump2.ecc[j]))


model.qc = Var(model.SI, within=NonNegativeReals, bounds=qc_bound)
model.qh = Var(model.SJ, within=NonNegativeReals, bounds=qh_bound)


def th_bound(m, i, k):
    return (float(heatpump2.thout[i]), float(heatpump2.thin[i]))


def tc_bound(m, j, k):
    return (float(heatpump2.tcin[j]), float(heatpump2.tcout[j]))


model.th = Var(model.SI, model.SK, within=NonNegativeReals, bounds=th_bound)
model.tc = Var(model.SJ, model.SK, within=NonNegativeReals, bounds=tc_bound)


def w_bound(m, p):
    return (0, float(heatpump2.wmax[p]))


model.w = Var(model.SP, within=NonNegativeReals, bounds=w_bound)


def dt_bound(m, i, j, k):
    return (float(heatpump2.EMAT), float(heatpump2.gamma[i, j]))


model.dt = Var(model.SI, model.SJ, model.SK, within=NonNegativeReals, bounds=dt_bound)


def dtcu_bound(m, i):
    return (float(heatpump2.EMAT), float(heatpump2.gammacu[i]))


def dthu_bound(m, j):
    return (float(heatpump2.EMAT), float(heatpump2.gammahu[j]))


def dthph_bound(m, i, p, k):
    return (float(heatpump2.EMAT), float(heatpump2.gammae[i, p, k]))


def dthpc_bound(m, j, k, p):
    return (float(heatpump2.EMAT), float(heatpump2.gammajk[j, k]))


model.dtcu = Var(model.SI, within=NonNegativeReals, bounds=dtcu_bound)
model.dthu = Var(model.SJ, within=NonNegativeReals, bounds=dthu_bound)
model.dthph = Var(model.SI, model.SP, model.SK, within=NonNegativeReals, bounds=dthph_bound)
model.dthpc = Var(model.SJ, model.SK, model.SP, within=NonNegativeReals, bounds=dthpc_bound)
model.dthpcs = Var(model.SJ, model.SK, model.SP, within=NonNegativeReals, bounds=dthpc_bound)

# binaries (on ST)
model.z = Var(model.SI, model.SJ, model.ST, within=Binary)
model.zhph = Var(model.SI, model.SP, model.ST, within=Binary)
model.zhpc = Var(model.SJ, model.SP, model.ST, within=Binary)
model.zhu = Var(model.SJ, within=Binary)
model.zcu = Var(model.SI, within=Binary)

# McCormick
def tca_bound(m, j, p, k):
    return (float(heatpump2.tcin[j]), float(heatpump2.tcout[j]))


model.tca = Var(model.SJ, model.SP, model.ST, within=NonNegativeReals, bounds=tca_bound)
model.w35_1 = Var(model.SJ, model.SP, model.ST, within=NonNegativeReals)
model.w35_2 = Var(model.SJ, model.SP, model.ST, within=NonNegativeReals)

model.fcpc = Var(model.SJ, model.SP, model.SK, within=NonNegativeReals, bounds=(10, 1000))
model.dtcj = Var(model.SJ, model.ST, within=NonNegativeReals)
model.y34 = Var(model.SJ, model.SP, model.ST, within=NonNegativeReals)

# area vars (LB objective里使用，若无面积下界约束会倾向0，这里先保留）
model.A_hx = Var(model.SI, model.SJ, model.ST, within=NonNegativeReals)
model.A_cu = Var(model.SI, within=NonNegativeReals)
model.A_hu = Var(model.SJ, within=NonNegativeReals)

# 常数
coef35 = {p: float(heatpump2.cphp[p] * (heatpump2.thpcin[p] - heatpump2.thpca[p])) for p in heatpump2.SP}
TCA_L = {j: float(heatpump2.tcin[j]) for j in heatpump2.SJ}
TCA_U = {j: float(heatpump2.tcout[j]) for j in heatpump2.SJ}
TC_L = {j: float(heatpump2.tcin[j]) for j in heatpump2.SJ}
TC_U = {j: float(heatpump2.tcout[j]) for j in heatpump2.SJ}
FCPC_L = 10.0
FCPC_U = 1000.0

DTL = {j: 0.0 for j in heatpump2.SJ}
DTU = {j: float(heatpump2.tcout[j] - heatpump2.tcin[j]) for j in heatpump2.SJ}
FL, FU = 10.0, 1000.0

# ============================================================
# 3) Base constraints
# ============================================================
def constrs_rule2(m, i):
    return (
        sum(m.q[i, j, k] for j in m.SJ for k in m.ST)
        + sum(m.qhph[i, p, k] for p in m.SP for k in m.ST)
        + m.qc[i]
        == heatpump2.fh[i] * (heatpump2.thin[i] - heatpump2.thout[i])
    )


def constrs_rule3(m, j):
    return (
        sum(m.q[i, j, k] for i in m.SI for k in m.ST)
        + sum(m.qhpc[j, p, k] for p in m.SP for k in m.ST)
        + m.qh[j]
        == heatpump2.fc[j] * (heatpump2.tcout[j] - heatpump2.tcin[j])
    )


def constrs_rule4(m, i, j, k):
    return m.q[i, j, k] <= m.z[i, j, k] * float(min(heatpump2.ech[i], heatpump2.ecc[j]))


def constrs_rule5(m, i):
    return m.qc[i] <= m.zcu[i] * float(heatpump2.ech[i])


def constrs_rule6(m, j):
    return m.qh[j] <= m.zhu[j] * float(heatpump2.ecc[j])


def constrs_rule7(m, j, p, k):
    return m.qhpc[j, p, k] <= m.zhpc[j, p, k] * float(heatpump2.ecc[j])


def constrs_rule8(m, i, p, k):
    return m.qhph[i, p, k] <= m.zhph[i, p, k] * float(heatpump2.ech[i])


def constrs_rule9(m, i, p):
    return sum(m.zhph[i, p, k] for k in m.ST) <= 1


def constrs_rule10(m, j, p):
    return sum(m.zhpc[j, p, k] for k in m.ST) <= 1


def constrs_rule11(m, p):
    return sum(m.zhph[i, p, k] for i in m.SI for k in m.ST) <= float(heatpump2.nbe[p])


def constrs_rule12(m, p):
    return sum(m.zhpc[j, p, k] for j in m.SJ for k in m.ST) <= float(heatpump2.nbc[p])


def constrs_rule13(m, i, p, k):
    return m.qhph[i, p, k] == float(heatpump2.evap[p]) * m.mhph[i, p, k] * (1 - float(heatpump2.hvap[p]))


def constrs_rule14(m, p):
    return m.mhp[p] == sum(m.mhph[i, p, k] for i in m.SI for k in m.ST)


def constrs_rule15(m, j, p, k):
    return m.qhpc[j, p, k] == (
        float(heatpump2.cond[p]) + float(heatpump2.cphp[p]) * (float(heatpump2.thpcin[p]) - float(heatpump2.thpca[p]))
    ) * m.mhpc[j, p, k]


def constrs_rule16(m, p):
    return m.mhp[p] == sum(m.mhpc[j, p, k] for j in m.SJ for k in m.ST)


def constrs_rule17(m, p):
    return sum(m.qhph[i, p, k] for i in m.SI for k in m.ST) + m.w[p] == \
        sum(m.qhpc[j, p, k] for j in m.SJ for k in m.ST)


def constrs_rule18(m, i, k):
    return sum(m.q[i, j, k] for j in m.SJ) + sum(m.qhph[i, p, k] for p in m.SP) == \
        heatpump2.fh[i] * (m.th[i, k] - m.th[i, k + 1])


def constrs_rule19(m, j, k):
    return sum(m.q[i, j, k] for i in m.SI) + sum(m.qhpc[j, p, k] for p in m.SP) == \
        heatpump2.fc[j] * (m.tc[j, k] - m.tc[j, k + 1])


def constrs_rule20(m, i):
    return m.th[i, 0] == float(heatpump2.thin[i])


def constrs_rule21(m, j):
    return m.tc[j, heatpump2.NK - 1] == float(heatpump2.tcin[j])


def constrs_rule22(m, i, k):
    return m.th[i, k] >= m.th[i, k + 1]


def constrs_rule23(m, i):
    return m.th[i, heatpump2.NK - 1] >= float(heatpump2.thout[i])


def constrs_rule24(m, j, k):
    return m.tc[j, k] >= m.tc[j, k + 1]


def constrs_rule25(m, j):
    return m.tc[j, 0] <= float(heatpump2.tcout[j])


def constrs_rule26(m, i):
    return m.qc[i] == heatpump2.fh[i] * (m.th[i, heatpump2.NK - 1] - float(heatpump2.thout[i]))


def constrs_rule27(m, j):
    return m.qh[j] == heatpump2.fc[j] * (float(heatpump2.tcout[j]) - m.tc[j, 0])


def constrs_rule28(m, i, j, k):
    return m.dt[i, j, k] <= m.th[i, k] - m.tc[j, k] + (1 - m.z[i, j, k]) * float(heatpump2.gamma[i, j])


def constrs_rule29(m, i, j, k):
    return m.dt[i, j, k + 1] <= m.th[i, k + 1] - m.tc[j, k + 1] + (1 - m.z[i, j, k]) * float(heatpump2.gamma[i, j])


def constrs_rule30(m, i):
    return m.dtcu[i] <= m.th[i, heatpump2.NK - 1] - float(heatpump2.tcuout) + (1 - m.zcu[i]) * float(heatpump2.gammacu[i])


def constrs_rule31(m, j):
    return m.dthu[j] <= float(heatpump2.thuout) - m.tc[j, 0] + (1 - m.zhu[j]) * float(heatpump2.gammahu[j])


def constrs_rule32(m, i, p, k):
    return m.dthph[i, p, k] <= m.th[i, k] - float(heatpump2.thpeout[p]) + (1 - m.zhph[i, p, k]) * float(heatpump2.gammae[i, p, k])


def constrs_rule33(m, i, p, k):
    return m.dthph[i, p, k + 1] <= m.th[i, k + 1] - float(heatpump2.thpein[p]) + (1 - m.zhph[i, p, k]) * float(heatpump2.gammae[i, p, k])


def constrs_rule36(m, j, k, p):
    return m.dthpc[j, k, p] <= float(heatpump2.thpcin[p]) - m.tc[j, k] + (1 - m.zhpc[j, p, k]) * float(heatpump2.gammajk[j, k])


def constrs_rule37(m, j, k, p):
    return m.dthpcs[j, k, p] <= float(heatpump2.thpca[p]) - m.tc[j, k] + (1 - m.zhpc[j, p, k]) * float(heatpump2.gammajk[j, k])


def constrs_rule38(m, j, k, p):
    return m.dthpc[j, k + 1, p] <= float(heatpump2.thpcout[p]) - m.tc[j, k + 1] + (1 - m.zhpc[j, p, k]) * float(heatpump2.gammajk[j, k])


def constrs_rule34_dtcj(m, j, k):
    return m.dtcj[j, k] == m.tc[j, k] - m.tc[j, k + 1]


def constrs_rule34_mcc1(m, j, p, k):
    xL = DTL[j]
    return m.y34[j, p, k] >= FL * m.dtcj[j, k] + xL * m.fcpc[j, p, k] - FL * xL


def constrs_rule34_mcc2(m, j, p, k):
    xU = DTU[j]
    return m.y34[j, p, k] >= FU * m.dtcj[j, k] + xU * m.fcpc[j, p, k] - FU * xU


def constrs_rule34_mcc3(m, j, p, k):
    xL = DTL[j]
    return m.y34[j, p, k] <= FU * m.dtcj[j, k] + xL * m.fcpc[j, p, k] - FU * xL


def constrs_rule34_mcc4(m, j, p, k):
    xU = DTU[j]
    return m.y34[j, p, k] <= FL * m.dtcj[j, k] + xU * m.fcpc[j, p, k] - FL * xU


def constrs_rule34_link(m, j, p, k):
    return m.qhpc[j, p, k] == m.y34[j, p, k]


def constrs_rule35_mcc1_1(m, j, p, k):
    return m.w35_1[j, p, k] >= TCA_L[j] * m.fcpc[j, p, k] + FCPC_L * m.tca[j, p, k] - TCA_L[j] * FCPC_L


def constrs_rule35_mcc1_2(m, j, p, k):
    return m.w35_1[j, p, k] >= TCA_U[j] * m.fcpc[j, p, k] + FCPC_U * m.tca[j, p, k] - TCA_U[j] * FCPC_U


def constrs_rule35_mcc1_3(m, j, p, k):
    return m.w35_1[j, p, k] <= TCA_U[j] * m.fcpc[j, p, k] + FCPC_L * m.tca[j, p, k] - TCA_U[j] * FCPC_L


def constrs_rule35_mcc1_4(m, j, p, k):
    return m.w35_1[j, p, k] <= TCA_L[j] * m.fcpc[j, p, k] + FCPC_U * m.tca[j, p, k] - TCA_L[j] * FCPC_U


def constrs_rule35_mcc2_1(m, j, p, k):
    return m.w35_2[j, p, k] >= TC_L[j] * m.fcpc[j, p, k] + FCPC_L * m.tc[j, k] - TC_L[j] * FCPC_L


def constrs_rule35_mcc2_2(m, j, p, k):
    return m.w35_2[j, p, k] >= TC_U[j] * m.fcpc[j, p, k] + FCPC_U * m.tc[j, k] - TC_U[j] * FCPC_U


def constrs_rule35_mcc2_3(m, j, p, k):
    return m.w35_2[j, p, k] <= TC_U[j] * m.fcpc[j, p, k] + FCPC_L * m.tc[j, k] - TC_U[j] * FCPC_L


def constrs_rule35_mcc2_4(m, j, p, k):
    return m.w35_2[j, p, k] <= TC_L[j] * m.fcpc[j, p, k] + FCPC_U * m.tc[j, k] - TC_L[j] * FCPC_U


def constrs_rule35_link(m, j, p, k):
    return m.w35_1[j, p, k] == m.w35_2[j, p, k] - coef35[p] * m.mhpc[j, p, k]


# Attach base constraints
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

# ============================================================
# 4) Uniqueness constraints (4 groups)
# ============================================================
st_list = sorted(list(model.ST))
st_prev = {st_list[t]: st_list[t - 1] for t in range(1, len(st_list))}
M_STAGE = len(heatpump2.SI) * len(heatpump2.SJ) + len(heatpump2.SI) * len(heatpump2.SP) + len(heatpump2.SJ) * len(heatpump2.SP)

# (1) no-empty-stage
def no_empty_stage_rule(m, k):
    if k not in st_prev:
        return Constraint.Skip
    kp = st_prev[k]
    cur = (
        sum(m.z[i, j, k] for i in m.SI for j in m.SJ)
        + sum(m.zhph[i, p, k] for i in m.SI for p in m.SP)
        + sum(m.zhpc[j, p, k] for j in m.SJ for p in m.SP)
    )
    prev = (
        sum(m.z[i, j, kp] for i in m.SI for j in m.SJ)
        + sum(m.zhph[i, p, kp] for i in m.SI for p in m.SP)
        + sum(m.zhpc[j, p, kp] for j in m.SJ for p in m.SP)
    )
    return cur - M_STAGE * prev <= 0
model.con_uniqueness_stage = Constraint(model.ST, rule=no_empty_stage_rule)

# (2) no-split for z(i,j,k)
def no_split_z_rule(m, i, j, k):
    if k not in st_prev:
        return Constraint.Skip
    kp = st_prev[k]
    prev_other = (
        sum(m.z[i, j1, kp] for j1 in m.SJ if j1 != j)
        + sum(m.zhph[i, p, kp] for p in m.SP)
        + sum(m.z[i1, j, kp] for i1 in m.SI if i1 != i)
        + sum(m.zhpc[j, p, kp] for p in m.SP)
    )
    cur_other = (
        sum(m.z[i, j1, k] for j1 in m.SJ if j1 != j)
        + sum(m.zhph[i, p, k] for p in m.SP)
        + sum(m.z[i1, j, k] for i1 in m.SI if i1 != i)
        + sum(m.zhpc[j, p, k] for p in m.SP)
    )
    return m.z[i, j, kp] + prev_other - m.z[i, j, k] + M_STAGE * cur_other >= 0
model.con_uniqueness_z = Constraint(model.SI, model.SJ, model.ST, rule=no_split_z_rule)

# (3) no-split for zhph(i,p,k)
def no_split_zhph_rule(m, i, p, k):
    if k not in st_prev:
        return Constraint.Skip
    kp = st_prev[k]
    prev_other = (
        sum(m.z[i, j, kp] for j in m.SJ)
        + sum(m.zhph[i, p1, kp] for p1 in m.SP if p1 != p)
        + sum(m.zhph[i1, p, kp] for i1 in m.SI if i1 != i)
    )
    cur_other = (
        sum(m.z[i, j, k] for j in m.SJ)
        + sum(m.zhph[i, p1, k] for p1 in m.SP if p1 != p)
        + sum(m.zhph[i1, p, k] for i1 in m.SI if i1 != i)
    )
    return m.zhph[i, p, kp] + prev_other - m.zhph[i, p, k] + M_STAGE * cur_other >= 0
model.con_uniqueness_zhph = Constraint(model.SI, model.SP, model.ST, rule=no_split_zhph_rule)

# (4) no-split for zhpc(j,p,k)
def no_split_zhpc_rule(m, j, p, k):
    if k not in st_prev:
        return Constraint.Skip
    kp = st_prev[k]
    prev_other = (
        sum(m.z[i, j, kp] for i in m.SI)
        + sum(m.zhpc[j, p1, kp] for p1 in m.SP if p1 != p)
        + sum(m.zhpc[j1, p, kp] for j1 in m.SJ if j1 != j)
    )
    cur_other = (
        sum(m.z[i, j, k] for i in m.SI)
        + sum(m.zhpc[j, p1, k] for p1 in m.SP if p1 != p)
        + sum(m.zhpc[j1, p, k] for j1 in m.SJ if j1 != j)
    )
    return m.zhpc[j, p, kp] + prev_other - m.zhpc[j, p, k] + M_STAGE * cur_other >= 0
model.con_uniqueness_zhpc = Constraint(model.SJ, model.SP, model.ST, rule=no_split_zhpc_rule)

# ============================================================
# 5) Objective (LB)
# ============================================================
C_HX = float(heatpump2.acoeff)
C_CU = float(heatpump2.cucoeff)
C_HU = float(heatpump2.hucoeff)

def obj_rule(m):
    utility = (
        sum(m.qh[j] * float(heatpump2.hucost) for j in m.SJ)
        + sum(m.qc[i] * float(heatpump2.cucost) for i in m.SI)
    )
    cap_area = (
        C_HX * sum(m.A_hx[i, j, k] for i in m.SI for j in m.SJ for k in m.ST)
        + C_CU * sum(m.A_cu[i] for i in m.SI)
        + C_HU * sum(m.A_hu[j] for j in m.SJ)
    )
    fixed_units = float(heatpump2.unitc) * (
        sum(m.z[i, j, k] for i in m.SI for j in m.SJ for k in m.ST)
        + sum(m.zcu[i] for i in m.SI)
        + sum(m.zhu[j] for j in m.SJ)
    )
    return utility + cap_area + fixed_units

model.obj = Objective(rule=obj_rule, sense=minimize)

# ============================================================
# 6) No-good enumeration machinery
# ============================================================
def extract_binary_pattern(m):
    bz = {(i, j, k): int(value(m.z[i, j, k]) > 0.5) for i in m.SI for j in m.SJ for k in m.ST}
    bzhu = {j: int(value(m.zhu[j]) > 0.5) for j in m.SJ}
    bzcu = {i: int(value(m.zcu[i]) > 0.5) for i in m.SI}
    bzhph = {(i, p, k): int(value(m.zhph[i, p, k]) > 0.5) for i in m.SI for p in m.SP for k in m.ST}
    bzhpc = {(j, p, k): int(value(m.zhpc[j, p, k]) > 0.5) for j in m.SJ for p in m.SP for k in m.ST}
    return bz, bzhu, bzcu, bzhph, bzhpc


def add_nogood_cut(m, patt):
    bz, bzhu, bzcu, bzhph, bzhpc = patt
    terms = []

    for i in m.SI:
        for j in m.SJ:
            for k in m.ST:
                b = bz[(i, j, k)]
                terms.append(m.z[i, j, k] if b == 1 else (1 - m.z[i, j, k]))

    for j in m.SJ:
        b = bzhu[j]
        terms.append(m.zhu[j] if b == 1 else (1 - m.zhu[j]))

    for i in m.SI:
        b = bzcu[i]
        terms.append(m.zcu[i] if b == 1 else (1 - m.zcu[i]))

    for i in m.SI:
        for p in m.SP:
            for k in m.ST:
                b = bzhph[(i, p, k)]
                terms.append(m.zhph[i, p, k] if b == 1 else (1 - m.zhph[i, p, k]))

    for j in m.SJ:
        for p in m.SP:
            for k in m.ST:
                b = bzhpc[(j, p, k)]
                terms.append(m.zhpc[j, p, k] if b == 1 else (1 - m.zhpc[j, p, k]))

    N = len(terms)
    return sum(terms) <= N - 1


model.ng_idx = Set(initialize=[])
model.NOGOOD = Constraint(model.ng_idx)

# ============================================================
# 7) Solve & enumerate
# ============================================================
solver = SolverFactory("glpk")
max_alt = 50
all_patterns = []

print("\n" + "=" * 60)
print("MILP-LB ENUMERATION RESULT")
print("=" * 60)

for alt in range(1, max_alt + 1):
    res = solver.solve(model, tee=False)
    term = str(res.solver.termination_condition).lower()

    if ("infeasible" in term) or ("unbounded" in term):
        print(f"\nStop at ALT={alt}: {res.solver.termination_condition}")
        break

    objv = float(value(model.obj))
    patt = extract_binary_pattern(model)
    all_patterns.append((objv, patt))

    print(f"\nALT={alt}, LB objective={objv:.6f}")

    bz, bzhu, bzcu, bzhph, bzhpc = patt

    print("Active z(i,j,k):")
    any_z = False
    for (i, j, k), b in bz.items():
        if b == 1:
            any_z = True
            print(f"  z[{i},{j},{k}] = 1")
    if not any_z:
        print("  (none)")

    print("Active zhu(j):", [j for j, b in bzhu.items() if b == 1] or "(none)")
    print("Active zcu(i):", [i for i, b in bzcu.items() if b == 1] or "(none)")

    print("Active zhph(i,p,k):")
    any_zhph = False
    for (i, p, k), b in bzhph.items():
        if b == 1:
            any_zhph = True
            print(f"  zhph[{i},{p},{k}] = 1")
    if not any_zhph:
        print("  (none)")

    print("Active zhpc(j,p,k):")
    any_zhpc = False
    for (j, p, k), b in bzhpc.items():
        if b == 1:
            any_zhpc = True
            print(f"  zhpc[{j},{p},{k}] = 1")
    if not any_zhpc:
        print("  (none)")

    # add no-good cut for next ALT
    model.ng_idx.add(alt)
    model.NOGOOD[alt] = add_nogood_cut(model, patt)

print(f"\nTotal structures found: {len(all_patterns)}")
print(f"Elapsed: {time.time() - start_time:.2f} s")