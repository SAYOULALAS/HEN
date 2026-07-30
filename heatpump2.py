import numpy as np

NI, NJ, NK ,NP= 2, 2, 3, 1

SI, SJ, SK, SP ,ST = range(NI), range(NJ), range(NK),range(NP), range(NK - 1)


# Hot stream
thin = np.array([465.0, 410.0])
thout = np.array([400.0, 310.0])
fh = np.array([80.1, 208.53])
hh = np.array([1.0, 1.0])


# Cold stream
tcin = np.array([315.0, 315.0])
tcout = np.array([370.0, 400.0])
fc = np.array([126.3, 213.6])
hc = np.array([1.0, 1.0])

# Costs and coefficients
hucost = 192.096;   hucoeff = 379.5; thuin = 416; thuout = 416; hhu = 1.0
cucost = 10.1952;   cucoeff = 379.5; tcuin = 290; tcuout = 300; hcu = 1.0
unitc = 0;  acoeff = 379.5
aexp = 0.65;      EMAT = 20
Tturout = 50.7; Tevain=40; Tconin=40; Tevaout=100; cpum=560; evacoeff =379.5
concoeff =379.5;
h1p = 5; h2p = 5;tebound =1000;cphp=5;teva=100;tcon=25
ELECTRICITY_COST = 150.0           # 电价 $/MWh
HP_FIXED_COST = 100000.0           # 每个热泵的固定投资成本
HP_EVAP_COEFF = 500.0              # 蒸发器换热器系数
HP_COND_COEFF = 500.0              # 冷凝器换热器系数
HP_AREA_EXP = 0.65                 # 面积指数
HP_UNIT_COST = 5000.0              # 每个换热器连接的固定成本
#evap=-0.00195*teva**2-0.4745*teva+203.24
#cond=-0.012836*tcon**2+1.688475*tcon+94.125
hvap=np.array([0.2,0.2])
evap=np.array([200,200])
cond=np.array([220,220])
nbe=np.array([2,2])
nbc=np.array([2,2])
cphp=np.array([5,5])
thpcin=np.array([450,460])
thpca=np.array([430,440])
thpcout=np.array([420,410])
wmax=np.array([10000,10000])
thpeout=np.array([350,340])
thpein=np.array([330,320])
dhp=np.array([160,160])

'''
NI, NJ, NK,NP = 3, 3, 3,2

SI, SJ, SK, ST,SP = range(NI), range(NJ), range(NK), range(NK - 1),range(NP)
thin = np.array([465.0, 410.0,454])
thout = np.array([400.0,300.0,433])
fh = np.array([44.5, 173.2,80])
hh = np.array([1.0, 1.0,1])


# Cold stream
tcin = np.array([293.0, 293.0,293])
tcout = np.array([398.0, 373,398])
fc = np.array([60.4, 52.6,160])
hc = np.array([1.0, 1.0,1])

# Costs and coefficients
hucost = 192.096;   hucoeff = 379.5; thuin = 416; thuout = 416; hhu = 1.0
cucost = 10.1952;   cucoeff = 379.5; tcuin = 290; tcuout = 300; hcu = 1.0
unitc = 0;  acoeff = 379.5
aexp = 0.65;      EMAT = 20
heva=1; hcon=1; hacu=1; Tconin=135.9; Tconout=80;
evacoeff =379.5
concoeff =379.5; regcoeff =379.5
ELECTRICITY_COST = 150.0           # 电价 $/MWh
HP_FIXED_COST = 100000.0           # 每个热泵的固定投资成本
HP_EVAP_COEFF = 500.0              # 蒸发器换热器系数
HP_COND_COEFF = 500.0              # 冷凝器换热器系数
HP_AREA_EXP = 0.65                 # 面积指数
HP_UNIT_COST = 5000.0              # 每个换热器连接的固定成本
hvap = np.array([0.05, 0.1])  # 干度
evap = np.array([1800, 1600])  # 蒸发潜热 (kJ/kg)
cond = np.array([850, 1000])   # 冷凝潜热 (kJ/kg)
nbe = np.array([2, 2])         # 最大蒸发器并联数
nbc = np.array([2, 2])         # 最大冷凝器并联数
cphp = np.array([6, 5.5])      # 比热容
thpcin = np.array([450, 470])  # 冷凝器入口温度 (K)
thpca = np.array([440, 460])   # 冷凝器饱和温度 (K)
thpcout = np.array([430, 450]) # 冷凝器出口温度 (K)
wmax = np.array([10000, 10000])  # 最大功耗限制 (kW)
thpeout = np.array([270, 280])  # 蒸发器出口温度 (K)
thpein = np.array([260, 270])   # 蒸发器入口温度 (K)
'''
'''
NI, NJ, NK ,NP= 2, 2, 3,2

SI, SJ, SK, ST,SP = range(NI), range(NJ), range(NK), range(NK - 1),range(NP)
thin = np.array([187.0, 127.0])
thout = np.array([77.0, 27.0])
fh = np.array([300.0, 500.0])
hh = np.array([1.0, 1.0])


# Cold stream
tcin = np.array([147.0, 47.0])
tcout = np.array([217.0, 117.0])
fc = np.array([600.0, 200.0])
hc = np.array([1.0, 1.0])

# Costs and coefficients
hucost = 192.096;   hucoeff = 379.5; thuin = 300; thuout = 250; hhu = 1.0
cucost = 10.1952;   cucoeff = 379.5; tcuin = 15; tcuout = 30; hcu = 1.0
unitc = 0;  acoeff = 379.5
aexp = 0.65;      EMAT = 10
heva=1; hcon=1; hacu=1; Tconin=60; Tconout=40; Tacuin=15; Tacuout=30; hregh=0.5; hregc=0.5
Tturout = 70; Tevain=40.2; Tevaout=87.5; cpum=560; evacoeff =379.5
concoeff =379.5; regcoeff =379.5

'''

ech = np.zeros(NI)
ech[SI] = fh[SI] * (thin[SI] - thout[SI])


ecc = np.zeros(NJ)
ecc[SJ] = fc[SJ] * (tcout[SJ] - tcin[SJ])


gamma = np.zeros((NI, NJ))
for i in SI:
    for j in SJ:
        gamma[i, j] = EMAT + max(abs(tcout[j] - thout[i]), abs(tcin [j] - thout[i]), abs(tcout[j] - thin[i]), abs(tcin[j] - thin[i]))
gammajk = np.zeros((NJ, NK))
for j in SJ:
    for k in SK:
        gammajk[j, k] = EMAT + 50
gammae = np.zeros((NI, NP,NK))
for i in SI:
    for p in SP:
        for k in SK:
            gammae[i, p, k] = EMAT + 50


gammacu = np.zeros(NI)




gammahu = np.zeros(NJ)



for i in SI:
    gammacu[i] = EMAT + max(abs(tcuout-thout[i]), abs(tcuout-thin[i]),
                            abs(tcuin-thout[i]),  abs(tcuin-thin[i]))

for j in SJ:
    gammahu[j] = EMAT + max(abs(tcout[j]-thuout), abs(tcout[j]-thuin),
                            abs(tcin[j]-thuout),  abs(tcin[j]-thuin))
Feasible_number = 0
for i in SI:
    for j in SJ:
        for k in ST:
            if thin[i] - tcin[j] >= EMAT:
                Feasible_number = Feasible_number + 1
print(Feasible_number)
print(ech)