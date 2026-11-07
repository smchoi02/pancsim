"""배경 사망: 곰페르츠 사망력 mu(a) = A*exp(B*a)를 2024 생명표 기대여명에 맞춘다.

근거: 통계청 2024 생명표 (헤럴드경제 2025.12 보도) 40세 기대여명 남 41.9/여 47.4년, 60세 남 23.7/여 28.4년.
당뇨 초과 사망은 나이 구간별 배수로 반영(Tancredi 2015 NEJM Table 3).
정식 분석에서는 KOSIS 생명표의 연령별 사망확률(qx)로 바꾸는 것을 권장한다(load_qx 참고).
"""
import numpy as np
from functools import lru_cache
from scipy.optimize import fsolve
from scipy.integrate import quad


def _surv(A, B, a, t):
    return np.exp(-A / B * (np.exp(B * (a + t)) - np.exp(B * a)))


def _ex(A, B, a, amax=110):
    return quad(lambda t: _surv(A, B, a, t), 0, amax - a)[0]


@lru_cache(maxsize=None)
def fit_gompertz(e40, e60):
    def eqs(x):
        A, B = np.exp(x)
        return [_ex(A, B, 40) - e40, _ex(A, B, 60) - e60]
    x = fsolve(eqs, [np.log(3e-5), np.log(0.09)])
    return tuple(np.exp(x))


def draw_death_age(age, male, U, p):
    """현재 나이 age에서 시작해 배경 사망까지의 시간(개월)을 역변환으로 뽑는다.

    당뇨 초과 사망: 일반인 곰페르츠 사망력에 '현재 나이' 구간별 배수를 곱한다
    (p["bg_mort_ages"] 경계, p["bg_mort_mult"] 배수; Tancredi 2015 NEJM Table 3).
    누적 사망력 H가 -ln U에 이를 때까지 나이 구간을 차례로 지나간다.
    """
    edges = list(p.get("bg_mort_ages", ())) + [np.inf]
    mults = list(p.get("bg_mort_mult", (1.0,)))
    lows = [-np.inf] + list(p.get("bg_mort_ages", ()))
    out = np.empty(len(age))
    for is_m, (e40, e60) in ((True, (p["e40_m"], p["e60_m"])), (False, (p["e40_f"], p["e60_f"]))):
        A, B = fit_gompertz(e40, e60)
        m = male == is_m
        a = age[m].astype(float).copy()
        E = -np.log(U[m])                       # 남은 누적 사망력
        death = np.full(m.sum(), np.nan)
        for lo, hi, k in zip(lows, edges, mults):
            act = np.isnan(death) & (a < hi)
            if not act.any():
                continue
            a0 = np.maximum(a[act], lo)
            seg = k * A / B * (np.exp(B * hi) - np.exp(B * a0)) if np.isfinite(hi) else np.full(act.sum(), np.inf)
            fin = E[act] <= seg
            idx = np.where(act)[0]
            # 이 구간에서 사망: exp(B·a_d) = exp(B·a0) + E·B/(k·A)
            d_age = np.log(np.exp(B * a0[fin]) + E[act][fin] * B / (k * A)) / B
            death[idx[fin]] = d_age
            nf = idx[~fin]
            E[nf] -= seg[~fin]
            a[nf] = hi
        out[m] = (death - age[m]) * 12
    return out


if __name__ == "__main__":
    import params
    p = params.get()
    for s, (e40, e60) in (("남", (p["e40_m"], p["e60_m"])), ("여", (p["e40_f"], p["e60_f"]))):
        A, B = fit_gompertz(e40, e60)
        print(s, "A=%.3e B=%.4f" % (A, B), "e40=%.1f e60=%.1f e70=%.1f" % (_ex(A, B, 40), _ex(A, B, 60), _ex(A, B, 70)),
              "1년 사망확률 60세 %.4f 70세 %.4f 80세 %.4f" % tuple(1 - _surv(A, B, a, 1) for a in (60, 70, 80)))
