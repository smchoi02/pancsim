"""자연경과 보정 (설계서 3.3절).

목표: (1) 신규 당뇨 발견 → 증상 진단 구간 분포 [Chari 2025 초록: 0-4, 4-12, 12-24, 24-36개월 = 30.5/31.3/19.5/18.7%]
      (2) 진단 병기 분포 [국가암등록 2017-2021: 국한 13.3, 국소 31.4, 원격 44.6% → 비례 재배분]
      (3) 국한→원격 약 16개월 [Yu 2015]
      (4) A형의 등록 시 단계 [46] [Pelaez-Luna 2007: 당뇨 진단 시점 CT 13명 중 정상 3, 절제 가능 6, 절제 불가 4]
          → P0/국한/국소 사전분포 Dirichlet(1+3, 1+6, 1+4) (균등 사전분포에 13명 자료를 더한 사후분포)
      0~4개월 구간은 허용 오차를 ±15%p로 넓힌다: Chari 코호트는 검사 기록의 혈당으로 신규 당뇨를 정의하므로
      췌장암 증상(황달·체중 감소)으로 검사받다가 혈당이 함께 잡힌 사람이 섞여 있고, 이들은 기기의 실제 사용자가 아니다.
이전 보정(A형 시작 분포 무정보, 0~4개월 ±5%p)은 calibrated_nh_old.json에 남겨 시나리오로 쓴다.
A형과 B형을 섞은 분포(가중치 = 3년 유병률 비)가 목표를 맞추도록 A형의 시작 분포와 공통 전이율을 찾는다.
실행: python calibrate.py [시도 횟수]  → calibrated_nh.json 저장 (채택된 모든 조합 + 기본 조합)
"""
import json
import numpy as np
from scipy.optimize import minimize
from natural_history import dx_distribution

T_TARGET = np.array([0.305, 0.313, 0.195, 0.187])
S_TARGET = np.array([13.3, 31.4, 44.6]); S_TARGET = S_TARGET / S_TARGET.sum()
YU_MONTHS = 15.6
TIMES = (4, 12, 24, 36)


def unpack(th):
    p0, pL, pR, hL, hR = np.exp(th[:5])
    hD = 0.2 + 0.8 / (1 + np.exp(-th[5]))           # 원격 전임상 1~5개월 (가정 범위)
    w = np.exp(np.r_[th[6], th[7], 0.0]); pi = w / w.sum()
    return dict(p0=p0, pL=pL, pR=pR, hL=hL, hR=hR, hD=hD), pi


def mixed_outputs(nh, piA, wA, wB, n_grid=37):
    cumA = dx_distribution(nh, np.r_[piA, 0], TIMES)                 # (4 times, 3 stages)
    # B형: onset ~ U(0,36), P0에서 시작 → 시각 t까지 진단 = onset이 t보다 앞선 경우의 (t-onset) 분포
    us = np.linspace(0, 36, n_grid)
    cumB = np.zeros_like(cumA)
    for i, t in enumerate(TIMES):
        acc = np.zeros(3)
        for u in us:
            if u < t:
                acc += dx_distribution(nh, np.array([1, 0, 0, 0.]), [t - u])[0]
        cumB[i] = acc / n_grid
    FA, FB = cumA.sum(1), cumB.sum(1)
    # 3년 안에 진단되는 사람만 '사례'로 정의 → 각 형을 3년 누적으로 정규화하고 유병률 비로 섞음
    TA = np.diff(np.r_[0, FA]) / FA[-1]; TB = np.diff(np.r_[0, FB]) / FB[-1]
    SA = cumA[-1] / FA[-1]; SB = cumB[-1] / FB[-1]
    T = wA * TA + wB * TB; S = wA * SA + wB * SB
    return T, S, FA[-1], FB[-1]



# ---------------------------------------------------------------------------
# 근사 베이지안 계산(ABC): 사전 범위에서 뽑고, 목표를 허용 오차 안에서 맞추는 조합만 채택.
# 자료만으로는 '등록 시 시작 분포'가 식별되지 않으므로(여러 조합이 똑같이 잘 맞음) 하나를 고르지 않고
# 채택된 조합 전체를 확률적 민감도 분석에 쓴다.
# 사전 범위 (근거): 국한·국소 진행 각 3~12개월, 합 10~20개월 (Yu 2015: 1기->4기 약 16개월),
#   원격 전임상 1~5개월(가정), 국한에서 증상 진단 비율 5~50%, 국소 10~70%(가정), P0 체류 3~36개월(가정),
#   A형 시작 분포 Dirichlet(4,7,5) (Pelaez-Luna 2007 자료 반영).
TOL_T = np.array([0.15, 0.05, 0.05, 0.05]); TOL_S = 0.03
PIA_ALPHA = (4, 7, 5)          # Pelaez-Luna 2007 (정상 3 / 절제 가능 6 / 절제 불가 4) + 균등 사전분포


def draw_prior(rng):
    pL = 1 / rng.uniform(3, 12); pR = 1 / rng.uniform(3, 12)
    if not 10 <= 1 / pL + 1 / pR <= 20:
        return None
    fL = rng.uniform(0.05, 0.5); fR = rng.uniform(0.10, 0.70)
    nh = dict(p0=1 / rng.uniform(3, 36), pL=pL, pR=pR,
              hL=fL / (1 - fL) * pL, hR=fR / (1 - fR) * pR, hD=1 / rng.uniform(1, 5))
    return nh, rng.dirichlet(PIA_ALPHA)


def abc(n_draws=40000, prev_A=0.0039, prev_B=0.0023, seed=1):
    wA, wB = prev_A / (prev_A + prev_B), prev_B / (prev_A + prev_B)
    rng = np.random.default_rng(seed)
    acc = []
    for _ in range(n_draws):
        d = draw_prior(rng)
        if d is None:
            continue
        nh, pi = d
        T, S, _, _ = mixed_outputs(nh, pi, wA, wB, n_grid=9)
        if np.all(np.abs(T - T_TARGET) <= TOL_T) and np.all(np.abs(S - S_TARGET) <= TOL_S):
            acc.append(dict(nh=nh, piA=pi.tolist(), err=float(np.sum((T - T_TARGET) ** 2) + np.sum((S - S_TARGET) ** 2))))
    return acc


def summarize(acc):
    keys = ["piP0", "piL", "piR", "dwellP0", "progL", "progR", "fracL", "fracR", "presympD"]
    rows = []
    for a in acc:
        nh, pi = a["nh"], a["piA"]
        rows.append([pi[0], pi[1], pi[2], 1 / nh["p0"], 1 / nh["pL"], 1 / nh["pR"],
                     nh["hL"] / (nh["hL"] + nh["pL"]), nh["hR"] / (nh["hR"] + nh["pR"]), 1 / nh["hD"]])
    X = np.array(rows)
    med = np.median(X, axis=0)
    # 기본 조합: 각 요약량의 중앙값에 (표준화 거리로) 가장 가까운 채택 조합
    z = (X - med) / (X.std(axis=0) + 1e-9)
    base = int(np.argmin((z ** 2).sum(1)))
    return keys, X, base


if __name__ == "__main__":
    import sys
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 40000
    acc = abc(n)
    keys, X, base = summarize(acc)
    print(f"채택 {len(acc)} / {n}")
    for k, col in zip(keys, X.T):
        print(f"{k:9s} 중앙값 {np.median(col):6.2f}  범위 {col.min():6.2f}~{col.max():6.2f}")
    out = dict(base=acc[base], accepted=acc)
    with open("calibrated_nh.json", "w") as f:
        json.dump(out, f, indent=1)
    print("기본 조합:", json.dumps(acc[base]))
