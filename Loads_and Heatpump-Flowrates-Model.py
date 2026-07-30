import numpy as np
from scipy.optimize import least_squares
import heatpump2

NI, NJ = len(heatpump2.SI), len(heatpump2.SJ)
NT = len(heatpump2.ST)
NP = len(heatpump2.SP)
NK = len(heatpump2.SK)

# =========================
# 已知量
# =========================
w = np.zeros(NP)
w[0] = 2000

qh = np.zeros(NJ)
qh[1] =2000


def unpack_x(x):
    idx = 0

    n_q = NI * NJ * NT
    q = x[idx:idx + n_q].reshape((NI, NJ, NT))
    idx += n_q

    n_mhp = NP
    mhp = x[idx:idx + n_mhp]
    idx += n_mhp

    n_mhph = NI * NP * NT
    mhph = x[idx:idx + n_mhph].reshape((NI, NP, NT))
    idx += n_mhph

    n_mhpc = NJ * NP * NT
    mhpc = x[idx:idx + n_mhpc].reshape((NJ, NP, NT))
    idx += n_mhpc

    n_qc = NI
    qc = x[idx:idx + n_qc]
    idx += n_qc

    n_qhph = NI * NP * NT
    qhph = x[idx:idx + n_qhph].reshape((NI, NP, NT))
    idx += n_qhph

    n_qhpc = NJ * NP * NT
    qhpc = x[idx:idx + n_qhpc].reshape((NJ, NP, NT))
    idx += n_qhpc

    return q,mhp,mhph,mhpc,qc, qhph, qhpc


def residuals(x):
    q,mhp,mhph,mhpc,qc, qhph, qhpc= unpack_x(x)

    res = []

    # Eq(2)
    for i in range(NI):
        lhs = sum(q[i, j, k] for j in range(NJ) for k in range(NT)) \
              + sum(qhph[i, p, k] for p in range(NP) for k in range(NT)) \
              + qc[i]
        rhs = heatpump2.fh[i] * (heatpump2.thin[i] - heatpump2.thout[i])
        res.append(lhs - rhs)

    # Eq(3)
    for j in range(NJ):
        lhs = sum(q[i, j, k] for i in range(NI) for k in range(NT)) \
              + sum(qhpc[j, p, k] for p in range(NP) for k in range(NT)) \
              + qh[j]
        rhs = heatpump2.fc[j] * (heatpump2.tcout[j] - heatpump2.tcin[j])
        res.append(lhs - rhs)

    for p in range(NP):
        lhs = sum(qhph[i, p, k] for i in range(NI) for k in range(NT)) + w[p]
        rhs = sum(qhpc[j, p, k] for j in range(NJ) for k in range(NT))
        res.append(lhs - rhs)

    for p in range(NP):
        rhs = mhp[p] * heatpump2.dhp[p]   #  dhp[p] 是已知常数
        res.append(w[p] - rhs)

    for j in range(NJ):
        for p in range(NP):
            coef = heatpump2.cond[p] + heatpump2.cphp[p] * (heatpump2.thpcin[p] - heatpump2.thpca[p])
            for k in range(NT):
                res.append(qhpc[j, p, k] - mhpc[j, p, k] * coef)
    # Eq(13)
    for i in range(NI):
        for p in range(NP):
            for k in range(NT):
                res.append(qhph[i, p, k] - heatpump2.evap[p] * mhph[i, p, k] * (1 - heatpump2.hvap[p]))

    # Eq(14)
    for p in range(NP):
        res.append(mhp[p] - sum(mhph[i, p, k] for i in range(NI) for k in range(NT)))

    # Eq(16)
    for p in range(NP):
        res.append(mhp[p] - sum(mhpc[j, p, k] for j in range(NJ) for k in range(NT)))

    return np.array(res)

# =========================
# 未知量维数与初始值
# =========================
n_q = NI * NJ * NT
n_mhp = NP
n_mhph = NI * NP * NT
n_mhpc = NJ * NP * NT
n_qc = NI
n_qhph = NI * NP * NT
n_qhpc = NJ * NP * NT

n_total = n_q + n_mhph + n_mhpc + n_mhp + n_qc + n_qhph + n_qhpc
x0 = np.ones(n_total)

# 设置初始值
ptr = 0
q0 = np.zeros((NI, NJ, NT))
q0[1, 1, 0] = 6342.0
q0[2, 0, 1] = 2208.0
x0[ptr:ptr + n_q] = q0.reshape(-1)
ptr += n_q

x0[ptr:ptr + n_mhp] = 20
ptr += n_mhp

mhph0 = np.zeros((NI, NP, NT))
#mhph0[1, 0, 1] = 46
x0[ptr:ptr + n_mhph] = mhph0.reshape(-1)
ptr += n_mhph

mhpc0 = np.zeros((NJ, NP, NT))
#mhpc0[0, 0, 1] = 6  # 对应保留的qhpc位置
x0[ptr:ptr + n_mhpc] = mhpc0.reshape(-1)
ptr += n_mhpc

qc0 = np.zeros(NI)
#qc0[1] = 8341.2
x0[ptr:ptr + n_qc] = qc0
ptr += n_qc

qhph0 = np.zeros((NI, NP, NT))
qhph0[0, 0, 1] = 2000.0
x0[ptr:ptr + n_qhph] = qhph0.reshape(-1)
ptr += n_qhph

qhpc0 = np.zeros((NJ, NP, NT))
qhpc0[2, 0, 0] = 16000.0
x0[ptr:ptr + n_qhpc] = qhpc0.reshape(-1)
ptr += n_qhpc

# =========================
# 边界约束设置（关键修复：避免lb=ub）
# =========================
lb = np.zeros(n_total)
ub = np.inf * np.ones(n_total)
FIXED_TOL = 1e-12  # 固定变量的极小上界，满足lb<ub

# 各变量起始索引
start_q = 0
start_mhp = start_q + n_q
start_mhph = start_mhp + n_mhp
start_mhpc = start_mhph + n_mhph
start_qc = start_mhpc + n_mhpc
start_qhph = start_qc + n_qc
start_qhpc = start_qhph + n_qhph

# q[0,1,0]、q[1,0,0]、q[1,1,1]
allow_q = {(1,1,0), (2,0,1)}
for i in range(NI):
    for j in range(NJ):
        for k in range(NT):
            pos = start_q + i * NJ * NT + j * NT + k
            if (i,j,k) not in allow_q:
                ub[pos] = FIXED_TOL  # 上界设为极小值，近似固定为0

# qhph[1,0,1]
allow_qhph = {(0,0,1)}
for i in range(NI):
    for p in range(NP):
        for k in range(NT):
            pos = start_qhph + i * NP * NT + p * NT + k
            if (i,p,k) not in allow_qhph:
                ub[pos] = FIXED_TOL

# 3. qhpc[0,0,1]
allow_qhpc = {(2,0,0)}
for j in range(NJ):
    for p in range(NP):
        for k in range(NT):
            pos = start_qhpc + j * NP * NT + p * NT + k
            if (j,p,k) not in allow_qhpc:
                ub[pos] = FIXED_TOL

# 4. qc[1]
allow_qc = {}
for i in range(NI):
    pos = start_qc + i
    if i not in allow_qc:
        ub[pos] = FIXED_TOL

# =========================
# 求解
# =========================
res_lsq = least_squares(residuals, x0, method='trf', bounds=(lb, ub), max_nfev=50000, verbose=2)
sol = res_lsq.x

# 把小于1e-10的变量强制设为0，消除数值误差
sol[sol < 1e-10] = 0

print("\n求解结果：")
print("success =", res_lsq.success)
print("message =", res_lsq.message)
print("residual norm =", np.linalg.norm(residuals(sol)))

q, mhp, mhph, mhpc, qc, qhph, qhpc = unpack_x(sol)
print("\nq =", q)
print("qhph =", qhph)
print("qhpc =", qhpc)
print("qc =", qc)
print("mhp =", mhp)
print("mhph =", mhph)
print("mhpc =", mhpc)