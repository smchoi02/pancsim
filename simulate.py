"""시뮬레이션 실행 (설계서 8장).

층화 표본: 대조군 n_ctrl명, A형 n_A명, B형 n_B명을 만들고, 결과를 유병률로 가중해 '신규 당뇨 1명당' 값으로 합친다.
학습용 코호트(다른 시드)로 판정 기준표를 만든 뒤, 시험용 코호트에서 평가한다(설계서 6.7절).
"""
import numpy as np
import pandas as pd
from cohort import make_cohort, age_risk_slope, CTRL, TYPE_A, TYPE_B
from strategies import STRATEGIES, LABELS, run_screening, build_reference
from outcomes import evaluate


def build_cohorts(p, nh, n_ctrl, n_A, n_B, seed):
    rng = np.random.default_rng(seed)
    beta = age_risk_slope(p, rng)
    return {CTRL: make_cohort(CTRL, n_ctrl, p, nh, rng, beta),
            TYPE_A: make_cohort(TYPE_A, n_A, p, nh, rng, beta),
            TYPE_B: make_cohort(TYPE_B, n_B, p, nh, rng, beta)}


def merge_cases(cA, cB):
    """기준표 학습용으로 A·B형을 유병률 비율대로 섞는다(단순 연결)."""
    out = {}
    for k in cA:
        if isinstance(cA[k], np.ndarray) and cA[k].ndim == 1 and len(cA[k]) == cA["n"]:
            out[k] = np.concatenate([cA[k], cB[k]])
    out["n"] = cA["n"] + cB["n"]; out["seed"] = cA["seed"]
    return out


def run_model(p, nh, n_ctrl=100_000, n_A=6_000, n_B=4_000, seed=2026, strategies=None, train_frac=0.5, verbose=True):
    """학습 코호트로 판정 모형을 만들고 시험 코호트에서 전략별 결과를 계산한다.
    반환: (전략별 요약 DataFrame, 전략별·집단별 사람 단위 결과 detail, 시험 코호트 test)"""
    strategies = strategies or list(STRATEGIES)
    pA = p["prev_total_3y"] - p["prev_B_3y"]; pB = p["prev_B_3y"]; pC = 1 - p["prev_total_3y"]
    W = {CTRL: pC, TYPE_A: pA, TYPE_B: pB}
    train = build_cohorts(p, nh, int(n_ctrl * train_frac), int(n_A * train_frac), int(n_B * train_frac), seed + 1)
    test = build_cohorts(p, nh, n_ctrl, n_A, n_B, seed)
    train_case = merge_cases(train[TYPE_A], train[TYPE_B])
    rows, detail = [], {}
    for s in strategies:
        kind = STRATEGIES[s]["kind"]
        ref = build_reference(train[CTRL], train_case, p, s)
        per = {}
        for ct, c in test.items():
            scr = run_screening(c, p, s, ref)
            per[ct] = evaluate(c, scr, p, s, kind)
        detail[s] = per
        agg = lambda key: sum(W[ct] * per[ct][key].mean() for ct in per)
        early = {ct: per[ct]["screen"].mean() for ct in (TYPE_A, TYPE_B)}
        rows.append(dict(전략=s, 이름=LABELS[s], 비용=agg("cost"), QALY=agg("qaly"), 생존연수=agg("ly"),
                         조기발견_전체=(pA * early[TYPE_A] + pB * early[TYPE_B]) / (pA + pB),
                         CT_만명당=1e4 * agg("ct"), 착용_1인당=agg("wear")))
        if verbose:
            r = rows[-1]
            print(f"{s} {LABELS[s]}: 비용 {r['비용']:.2f}만원, QALY {r['QALY']:.5f}, 조기발견 {r['조기발견_전체']:.3f}", flush=True)
    return pd.DataFrame(rows).set_index("전략"), detail, test
