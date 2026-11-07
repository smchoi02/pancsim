"""췌장암 자연경과 (설계서 3.2~3.3절).

상태: 0 = P0(미세), 1 = L(국한), 2 = R(국소), 3 = D(원격). 시간 단위는 개월.
A형: 등록 시 P0/L/R 중 하나에서 시작. B형: 등록 후 onset 시점에 P0에서 시작.
매 상태에서 '다음 병기로 진행'과 '증상으로 진단'이 경쟁한다(지수분포 대기시간).
"""
import numpy as np
from scipy.linalg import expm

STATE_NAMES = ["P0", "L", "R", "D"]


def generator(nh):
    """연속시간 마르코프 생성행렬. 상태 0~3(전임상), 4~6(L/R/D로 진단됨, 흡수)."""
    p0, pL, pR, hL, hR, hD = (nh[k] for k in ("p0", "pL", "pR", "hL", "hR", "hD"))
    Q = np.zeros((7, 7))
    Q[0, 1] = p0
    Q[1, 2] = pL; Q[1, 4] = hL
    Q[2, 3] = pR; Q[2, 5] = hR
    Q[3, 6] = hD
    for i in range(4):
        Q[i, i] = -Q[i].sum()
    return Q


def dx_distribution(nh, start, times):
    """start(길이 4 시작 분포)에서 출발했을 때 각 시점까지 진단된 누적 확률 (L,R,D별)."""
    Q = generator(nh)
    s = np.r_[start, 0, 0, 0]
    return np.array([s @ expm(Q * t) for t in times])[:, 4:]


def simulate_paths(nh, n, rng, start_probs=None, onset=None):
    """개인별 경로를 생성한다.

    반환: dict(tL, tR, tD, Tc, stage_c) — 각 상태 진입 시각과 증상 진단 시각·병기(1=L,2=R,3=D).
    도달하지 않은 상태는 inf. onset이 주어지면(B형) 그 시각에 P0에서 시작.
    """
    p0, pL, pR, hL, hR, hD = (nh[k] for k in ("p0", "pL", "pR", "hL", "hR", "hD"))
    if onset is None:
        start = rng.choice(3, size=n, p=start_probs)   # 0=P0, 1=L, 2=R
        t0 = np.zeros(n)
    else:
        start = np.zeros(n, dtype=int)
        t0 = np.asarray(onset, dtype=float)
    tL = np.full(n, np.inf); tR = np.full(n, np.inf); tD = np.full(n, np.inf)
    Tc = np.full(n, np.inf); stage = np.zeros(n, dtype=int)
    t = t0.copy()
    # P0 -> L
    m = start == 0
    t[m] += rng.exponential(1 / p0, m.sum())
    inL = start <= 1
    tL[inL] = t[inL]
    # L: 진행 vs 진단
    dwell = rng.exponential(1 / (pL + hL), n)
    toR = rng.random(n) < pL / (pL + hL)
    t[inL] += dwell[inL]
    dxL = inL & ~toR
    Tc[dxL] = t[dxL]; stage[dxL] = 1
    inR = (inL & toR) | (start == 2)
    tR[inR] = t[inR]
    if onset is None:
        tR[start == 2] = 0.0
    dwell = rng.exponential(1 / (pR + hR), n)
    toD = rng.random(n) < pR / (pR + hR)
    t[inR] += dwell[inR]
    dxR = inR & ~toD
    Tc[dxR] = t[dxR]; stage[dxR] = 2
    inD = inR & toD
    tD[inD] = t[inD]
    t[inD] += rng.exponential(1 / hD, inD.sum())
    Tc[inD] = t[inD]; stage[inD] = 3
    return dict(tL=tL, tR=tR, tD=tD, Tc=Tc, stage_c=stage)


def state_at(paths, t):
    """시각 t(개월)의 숨은 상태. -1 = 아직 종양 없음(B형 onset 전) 또는 비암, 0~3 = P0~D."""
    tL, tR, tD = paths["tL"], paths["tR"], paths["tD"]
    onset = paths.get("onset")
    s = np.zeros_like(tL, dtype=int)
    s[t >= tL] = 1
    s[t >= tR] = 2
    s[t >= tD] = 3
    if onset is not None:
        s[t < onset] = -1
    return s
