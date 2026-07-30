import sympy as sp

NI, NJ, NK, NP = 2, 2, 3, 1
SI, SJ, SK, SP, ST = range(NI), range(NJ), range(NK), range(NP), range(NK - 1)

thin  = [465.0, 410.0]
thout = [400.0, 310.0]
fh    = [80.1, 208.53]
tcin  = [315.0, 315.0]
tcout = [370.0, 400.0]
fc    = [126.3, 213.6]

# =============================
# 结果 — 激活变量
# =============================
z_active    = {(0, 1, 0): 1.0, (1, 0, 0): 1.0}
zhph_active = {(1, 0, 1): 1.0}
zhpc_active = {(0, 0, 0): 1.0, (1, 0, 0): 1.0}
zcu_active  = {1: 1.0}
zhu_active  = {1: 1.0}

# =============================
# 变量
# =============================
q = {}
for i in SI:
    for j in SJ:
        for k in ST:
            q[(i, j, k)] = (
                sp.symbols(f"q_{i}_{j}_{k}")
                if (i, j, k) in z_active
                else sp.Integer(0)
            )

qhph = {}
for i in SI:
    for p in SP:
        for k in ST:
            qhph[(i, p, k)] = (
                sp.symbols(f"qhph_{i}_{p}_{k}")
                if (i, p, k) in zhph_active
                else sp.Integer(0)
            )

qhpc = {}
for j in SJ:
    for p in SP:
        for k in ST:
            qhpc[(j, p, k)] = (
                sp.symbols(f"qhpc_{j}_{p}_{k}")
                if (j, p, k) in zhpc_active
                else sp.Integer(0)
            )

qc = {}
for i in SI:
    qc[i] = sp.symbols(f"qc_{i}") if i in zcu_active else sp.Integer(0)

qh = {}
for j in SJ:
    qh[j] = sp.symbols(f"qh_{j}") if j in zhu_active else sp.Integer(0)

th = {(i, k): sp.symbols(f"th_{i}_{k}") for i in SI for k in SK}
tc = {(j, k): sp.symbols(f"tc_{j}_{k}") for j in SJ for k in SK}

# =============================
# 构建方程
# =============================
eqs = []

# constrs2 — 热流总体平衡 (SI)
for i in SI:
    lhs = (
        sum(q[(i, j, k)] for j in SJ for k in ST)
        + sum(qhph[(i, p, k)] for p in SP for k in ST)
        + qc[i]
    )
    eqs.append(sp.Eq(lhs, fh[i] * (thin[i] - thout[i])))

# constrs3 — 冷流总体平衡 (SJ)
for j in SJ:
    lhs = (
        sum(q[(i, j, k)] for i in SI for k in ST)
        + sum(qhpc[(j, p, k)] for p in SP for k in ST)
        + qh[j]
    )
    eqs.append(sp.Eq(lhs, fc[j] * (tcout[j] - tcin[j])))

# constrs18 — 热流级内平衡 (SI x ST)
for i in SI:
    for k in ST:
        lhs = sum(q[(i, j, k)] for j in SJ) + sum(qhph[(i, p, k)] for p in SP)
        eqs.append(sp.Eq(lhs, fh[i] * (th[(i, k)] - th[(i, k + 1)])))

# constrs19 — 冷流级内平衡 (SJ x ST)
for j in SJ:
    for k in ST:
        lhs = sum(q[(i, j, k)] for i in SI) + sum(qhpc[(j, p, k)] for p in SP)
        eqs.append(sp.Eq(lhs, fc[j] * (tc[(j, k)] - tc[(j, k + 1)])))

# constrs40 — 热流公用工程 (SI)
for i in SI:
    eqs.append(sp.Eq(qc[i], fh[i] * (th[(i, 2)] - thout[i])))

# constrs41 — 冷流公用工程 (SJ)
for j in SJ:
    eqs.append(sp.Eq(qh[j], fc[j] * (tcout[j] - tc[(j, 0)])))

for i in SI:
    eqs.append(sp.Eq(th[(i, 0)], thin[i]))
#for i in SI:
#    eqs.append(sp.Eq(th[(i, 2)], thout[i]))
for j in SJ:
    eqs.append(sp.Eq(tc[(j, 0)], tcout[j]))
#for j in SJ:
#    eqs.append(sp.Eq(tc[(j, 2)], tcin[j]))

# =============================
# 变量与方程统计
# =============================
all_vars = []
for container in (q, qhph, qhpc, qc, qh, th, tc):
    for v in container.values():
        if isinstance(v, sp.Symbol):
            all_vars.append(v)
all_vars = sorted(set(all_vars), key=str)

n_var = len(all_vars)
n_eq = len(eqs)

labels = (
    [f"constrs2(i={i})" for i in SI]
    + [f"constrs3(j={j})" for j in SJ]
    + [f"constrs18(i={i},k={k})" for i in SI for k in ST]
    + [f"constrs19(j={j},k={k})" for j in SJ for k in ST]
    + [f"constrs40(i={i})" for i in SI]
    + [f"constrs41(j={j})" for j in SJ]
    + [f"bc_th(i={i})" for i in SI]
    + [f"bc_tc(j={j})" for j in SJ]
)

print(f"  方程明细")

for name, eq in zip(labels, eqs):
    diff = sp.simplify(eq.lhs - eq.rhs)
    if diff != 0:
        print(f"    {name}:  {eq.lhs} = {eq.rhs}")
print(f"\n  方程总数 N_eq = {n_eq}")

# =============================
# 构建系数矩阵 A
# =============================
A = sp.Matrix.zeros(n_eq, n_var)
for eq_idx, eq in enumerate(eqs):
    expr = sp.expand(eq.lhs - eq.rhs)
    for var_idx, var in enumerate(all_vars):
        coeff = expr.coeff(var)
        if coeff != 0:
            A[eq_idx, var_idx] = coeff

rank = A.rank()

# =============================

dof = n_var - rank
print(f"  独立变量数       = {n_var}")
print(f"  独立方程数       = {rank}")
print(f"  自由度           = {n_var} - {rank} = {dof}")



