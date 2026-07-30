import numpy as np
from scipy.optimize import least_squares
import heatpump2
import math
NI, NJ = len(heatpump2.SI), len(heatpump2.SJ)
NT = len(heatpump2.ST)
NP = len(heatpump2.SP)
NK = len(heatpump2.SK)
# =========================
# 已知量
# =========================
w = np.zeros(NP)
w[0] = 7384.2
qh = np.zeros(NJ)
q = np.zeros((NI, NJ, NT))
q[0, 1, 0] = 5206.5
q[1, 0, 0] = 5127.6
qhph = np.zeros((NI, NP, NT))
qhph[1, 0, 1] = 7384.2
qhpc = np.zeros((NJ, NP, NT))
qhpc[0, 0, 0] = 1818.9
qhpc[1, 0, 0] = 12949.5
mhph = np.zeros((NI, NP, NT))
mhph[1, 0, 1] = 46.15
mhpc = np.zeros((NJ, NP, NT))
mhpc[0, 0, 0] = 5.68
mhpc[1, 0, 0] = 40.47
mhp = np.zeros(NP)
mhp[0] = 46.15
qc = np.zeros(NI)
qc[1] = 8341.19
qh = np.zeros(NJ)
PENALTY_W = 1e3  #
def unpack_x(x):
    idx = 0
    n_th = NI * NK
    th = x[idx:idx + n_th].reshape((NI, NK))
    idx += n_th
    n_tc = NJ * NK
    tc = x[idx:idx + n_tc].reshape((NJ, NK))
    idx += n_tc
    n_fcpc = NJ * NP * NT
    fcpc = x[idx:idx + n_fcpc].reshape((NJ, NP, NT))
    idx += n_fcpc
    n_tca = NJ * NP * NT
    tca = x[idx:idx + n_tca].reshape((NJ, NP, NT))
    idx += n_tca
    return th, tc, fcpc, tca
def residuals(x):
    th, tc, fcpc, tca = unpack_x(x)
    res = []
    # Eq(18)
    for i in range(NI):
        for k in range(NT):
            lhs = sum(q[i, j, k] for j in range(NJ)) + sum(qhph[i, p, k] for p in range(NP))
            rhs = heatpump2.fh[i] * (th[i, k] - th[i, k + 1])
            res.append(lhs - rhs)
    # Eq(19)
    for j in range(NJ):
        for k in range(NT):
            lhs = sum(q[i, j, k] for i in range(NI)) + sum(qhpc[j, p, k] for p in range(NP))
            rhs = heatpump2.fc[j] * (tc[j, k] - tc[j, k + 1])
            res.append(lhs - rhs)
    # Eq(20)
    for i in range(NI):
        res.append(th[i, 0] - heatpump2.thin[i])
    # Eq(21)
    for j in range(NJ):
        res.append(tc[j, NK - 1] - heatpump2.tcin[j])
    # Eq(26)
    for i in range(NI):
        res.append(qc[i] - heatpump2.fh[i] * (th[i, NK - 1] - heatpump2.thout[i]))
    # Eq(27)
    for j in range(NJ):
        res.append(qh[j] - heatpump2.fc[j] * (heatpump2.tcout[j] - tc[j, 0]))
    # Eq(34)
    for j in range(NJ):
        for p in range(NP):
            for k in range(NT):
                res.append(qhpc[j, p, k] - fcpc[j, p, k] * (tc[j, k] - tc[j, k + 1]))
    # Eq(35)
    for j in range(NJ):
        for p in range(NP):
            cterm = heatpump2.cphp[p] * (heatpump2.thpcin[p] - heatpump2.thpca[p])
            for k in range(NT):
                rhs = tc[j, k] - (mhpc[j, p, k] * cterm) * (1.0 / fcpc[j, p, k])
                res.append(tca[j, p, k] - rhs)
    # (22) th[i,k] >= th[i,k+1] max(0.0, vio)：仅对违反情况生效 PENALTY_W：惩罚权重
    for i in range(NI):
        for k in range(NK - 1):
            vio = th[i, k + 1] - th[i, k]      # >0 表示违反
            res.append(PENALTY_W * max(0.0, vio))

    # (23) th[i,NK-1] >= thout[i]
    for i in range(NI):
        vio = heatpump2.thout[i] - th[i, NK - 1]
        res.append(PENALTY_W * max(0.0, vio))

    # (24) tc[j,k] >= tc[j,k+1]
    for j in range(NJ):
        for k in range(NK - 1):
            vio = tc[j, k + 1] - tc[j, k]
            res.append(PENALTY_W * max(0.0, vio))

    # (25) tc[j,0] <= tcout[j]
    for j in range(NJ):
        vio = tc[j, 0] - heatpump2.tcout[j]
        res.append(PENALTY_W * max(0.0, vio))

    return np.array(res)
# =========================
# 未知量维数（已去掉q/qhph/qhpc）
# =========================
n_th = NI * NK
n_tc = NJ * NK
n_fcpc = NJ * NP * NT
n_tca = NJ * NP * NT
n_total =  n_th + n_tc + n_fcpc + n_tca
x0 = np.ones(n_total)
ptr = 0
th0 = np.zeros((NI, NK))
for i in range(NI):
    th0[i, :] = np.linspace(heatpump2.thin[i], heatpump2.thout[i], NK)
x0[ptr:ptr+n_th] = th0.reshape(-1); ptr += n_th
tc0 = np.zeros((NJ, NK))
for j in range(NJ):
    tc0[j, :] = np.linspace(heatpump2.tcout[j], heatpump2.tcin[j], NK)
x0[ptr:ptr+n_tc] = tc0.reshape(-1); ptr += n_tc
fcpc0 = np.zeros((NJ, NP, NT))
fcpc0[0, 0, 0] = 33; fcpc0[0, 0, 1] = 10
fcpc0[1, 0, 0] = 150; fcpc0[1, 0, 1] = 10
x0[ptr:ptr+n_fcpc] = fcpc0.reshape(-1); ptr += n_fcpc
tca0 = np.zeros((NJ, NP, NT))
tca0[0, 0, 0] = 352; tca0[0, 0, 1] = 315
tca0[1, 0, 0] = 373; tca0[1, 0, 1] = 315
x0[ptr:ptr+n_tca] = tca0.reshape(-1); ptr += n_tca
res_lsq = least_squares(residuals, x0, method='trf', max_nfev=20000, verbose=2)
sol = res_lsq.x
print("success =", res_lsq.success)
print("message =", res_lsq.message)
print("residual norm =", np.linalg.norm(residuals(sol)))
th, tc, fcpc, tca = unpack_x(sol)
print("th =", th)
print("tc =", tc)
print("fcpc =", fcpc)
print("tca =", tca)

# =========================
# 【新增】换热面积计算模块
# =========================
# 预计算总传热系数（与你之前的模型保持一致）
Uij = {(i,j): 1.0/(1.0/heatpump2.hh[i] + 1.0/heatpump2.hc[j]) for i in range(NI) for j in range(NJ)}
Ucu = {i: 1.0/(1.0/heatpump2.hh[i] + 1.0/heatpump2.hcu) for i in range(NI)}
Uhu = {j: 1.0/(1.0/heatpump2.hc[j] + 1.0/heatpump2.hhu) for j in range(NJ)}

# 公用工程参数
tcu_in = heatpump2.tcuin    # 冷公用工程入口温度 如30℃
tcu_out = heatpump2.tcuout  # 冷公用工程出口温度 如40℃
thu_in = heatpump2.thuin    # 热公用工程入口温度 如160℃
thu_out = heatpump2.thuout  # 热公用工程出口温度 如160℃
# 对数平均温差计算工具函数

def calc_lmtd(dT1, dT2):
    if dT1 < 1e-3 or dT2 < 1e-3:
        return 1e-3
    if abs(dT1 - dT2) < 1e-3:
        return (dT1 + dT2) / 2
    return (dT1 - dT2) / math.log(dT1 / dT2)
# 1.常规换热器面积计算
A_hx = np.zeros_like(q)
for i in range(NI):
    for j in range(NJ):
        for k in range(NT):
            if q[i,j,k] > 1e-3:  # 仅计算存在换热的匹配
                dT1 = th[i,k] - tc[j,k]    # 热端温差（逆流）
                dT2 = th[i,k+1] - tc[j,k+1]# 冷端温差
                lmtd = calc_lmtd(dT1, dT2)
                A_hx[i,j,k] = q[i,j,k] / (Uij[(i,j)] * lmtd)
# 2.冷却器面积计算
A_cu = np.zeros_like(qc)
for i in range(NI):
    if qc[i] > 1e-3:
        th_in = th[i, -1] #th(i,lastk)
        th_out = heatpump2.thout[i]
        dT1 = th_in - tcu_out
        dT2 = th_out - tcu_in
        lmtd = calc_lmtd(dT1, dT2)
        A_cu[i] = qc[i] / (Ucu[i] * lmtd)
#3.加热器面积计算
A_hu = np.zeros_like(qh)
for j in range(NJ):
    if qh[j] > 1e-3:
        tc_in = tc[j, 0]
        tc_out = heatpump2.tcout[j]
        dT1 = thu_in - tc_out
        dT2 = thu_out - tc_in
        lmtd = calc_lmtd(dT1, dT2)
        A_hu[j] = qh[j] / (Uhu[j] * lmtd)

# 打印面积结果
print("\n" + "="*60)
print("换热面积计算结果")
print("="*60)
print("常规换热器面积(m²):")
for i in range(NI):
    for j in range(NJ):
        for k in range(NT):
            if A_hx[i,j,k] > 1e-3:
                print(f"  热{i}-冷{j}-阶段{k}: {A_hx[i,j,k]:.1f}")
print("\n冷却器面积(m²):")
for i in range(NI):
    if A_cu[i] > 1e-3:
        print(f"  热{i}: {A_cu[i]:.1f}")

print("\n加热器面积(m²):")
for j in range(NJ):
    if A_hu[j] > 1e-3:
        print(f"  冷{j}: {A_hu[j]:.1f}")

# 总面积统计
total_A = np.sum(A_hx) + np.sum(A_cu) + np.sum(A_hu)

print(f"\n总换热面积: {total_A:.1f} m²")

eps = 1e-3  # 判定阈值，避免数值噪声

# 常规换热器存在性 z_hx[i,j,k] 将布尔值转换为浮点数（True → 1.0，False → 0.0）
z = (q > eps).astype(float)      # shape (NI, NJ, NT)

# 加热器存在性 z_hu[j]
zhu = (qh > eps).astype(float)     # shape (NJ,)

# 冷却器存在性 z_cu[i]
zcu = (qc > eps).astype(float)     # shape (NI,)

FCEX = heatpump2.unitc * (np.sum(z)+np.sum(zhu)+np.sum(zcu))

C_HX = heatpump2.acoeff    # 常规换热器单位面积成本 元/m²
C_CU = heatpump2.cucoeff   # 冷却器单位面积成本 元/m²
C_HU = heatpump2.hucoeff   # 加热器单位面积成本 元/m²

hucost = heatpump2.hucost  # 热公用工程成本 元/kWh
cucost = heatpump2.cucost  # 冷公用工程成本 元/kWh

# 1. 设备投资成本
cap_cost = (np.sum(A_hx)**heatpump2.aexp * C_HX +
            np.sum(A_cu)**heatpump2.aexp * C_CU +
            np.sum(A_hu)**heatpump2.aexp * C_HU
            )
# 2. 年公用工程成本
utility_cost = (np.sum(qh) * hucost +
                np.sum(qc) * cucost
                )
# 3. 总年化成本
TAC = cap_cost  + utility_cost + FCEX
# 打印TAC结果
print("\n" + "="*60)
print("TAC计算结果")
print("="*60)
print(f"设备总投资: {cap_cost/1e4:.2f} 万元")
print(f"总年化成本TAC: {TAC/1e4:.2f} 万元/年")