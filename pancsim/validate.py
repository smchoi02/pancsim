"""모델 검증 (설계서 9장): 모델이 만든 값이 실제 자료와 맞는지 확인한다.

1) 진단 시점 분포 (일반 진료)  vs  Chari 2025 [1] 0-4/4-12/12-24/24-36개월 = 30.5/31.3/19.5/18.7%
2) 진단 병기 분포 (일반 진료)   vs  국가암등록 [2] 국한 14.9 / 국소 35.2 / 원격 50.0% (병기 미상 제외)
3) 진단 후 5년 생존             vs  국가암등록 [2] 췌장암 5년 상대생존 약 16%
4) ENDPAC 성능 (S1)             vs  Sharma 2018 [11] 검증 코호트 민감도 78%, 특이도 82%(미국)
                                     / 영국 외부 검증의 특이도 약 72% [28]
"""
import numpy as np
import pandas as pd
from cohort import CTRL, TYPE_A, TYPE_B
from strategies import endpac_score
from outcomes import survival_months

T_TARGET = (0.305, 0.313, 0.195, 0.187)
S_TARGET = (0.149, 0.352, 0.500)


def validation_table(p, test, detail):
    pA = p["prev_total_3y"] - p["prev_B_3y"]; pB = p["prev_B_3y"]; pC = 1 - p["prev_total_3y"]
    cA, cB, cC = test[TYPE_A], test[TYPE_B], test[CTRL]
    wA = np.full(cA["n"], pA / cA["n"]); wB = np.full(cB["n"], pB / cB["n"])
    rows = []

    # 1) 등록 후 진단 시점 분포 (B형은 등록 후 개월로 이미 표현됨)
    Tc = np.concatenate([cA["Tc"], cB["Tc"]]); w = np.concatenate([wA, wB])
    edges = [0, 4, 12, 24, 36.0001]
    for i, lab in enumerate(["0~4개월", "4~12개월", "12~24개월", "24~36개월"]):
        m = (Tc >= edges[i]) & (Tc < edges[i + 1])
        rows.append(("진단 시점", lab, w[m].sum() / w.sum(), T_TARGET[i], "Chari 2025 [1] 초록"))

    # 2) 일반 진료(S0)에서의 진단 병기
    st = np.concatenate([cA["stage_c"], cB["stage_c"]])
    for lab, ss, tgt in (("국한(P0 포함)", [0, 1], S_TARGET[0]), ("국소", [2], S_TARGET[1]), ("원격", [3], S_TARGET[2])):
        rows.append(("진단 병기(S0)", lab, w[np.isin(st, ss)].sum() / w.sum(), tgt, "국가암등록통계 [2]"))

    # 3) 5년 생존 (암 생존만, 상대생존에 대응)
    surv = survival_months(st, np.concatenate([cA["U_surv"], cB["U_surv"]]), p)
    rows.append(("생존", "진단 후 5년 생존(S0)", w[surv > 60].sum() / w.sum(), 0.16, "국가암등록통계 [2]"))
    for lab, ss, tgt in (("국한", [0, 1], 0.472), ("국소", [2], np.nan), ("원격", [3], np.nan)):
        m = np.isin(st, ss)
        rows.append(("생존", f"5년 생존 - {lab}", w[m & (surv > 60)].sum() / max(w[m].sum(), 1e-12), tgt, "[2] (국한 47.2%)"))

    # 4) ENDPAC (점수 계산 가능한 사람 기준)
    def pos_rate(c, wt):
        ok = c["endpac_ok"]
        return (wt[ok] * (endpac_score(c)[ok] >= 3)).sum() / wt[ok].sum()
    sens = (pA * pos_rate(cA, wA / wA.sum()) + pB * pos_rate(cB, wB / wB.sum())) / (pA + pB)
    sensA = pos_rate(cA, wA / wA.sum())
    spec = 1 - pos_rate(cC, np.ones(cC["n"]))
    ppv = sens * (pA + pB) / (sens * (pA + pB) + (1 - spec) * pC)
    rows += [("ENDPAC", "민감도 - A형(등록 시 이미 암)", sensA, 0.78, "Sharma 2018 [11] Table 2 검증"),
             ("ENDPAC", "민감도 - A+B형 전체", sens, np.nan, "B형은 등록 때 신호 없음"),
             ("ENDPAC", "특이도", spec, 0.72, "영국 외부검증 [28] (미국 원논문 0.82)"),
             ("ENDPAC", "양성예측도", ppv, np.nan, "")]

    # 5) 전략별 실제 정밀검사 결과 요약
    if "S1" in detail:
        d = detail["S1"]
        ct = sum(W * d[k]["ct"].mean() for k, W in ((CTRL, pC), (TYPE_A, pA), (TYPE_B, pB)))
        found = pA * d[TYPE_A]["screen"].mean() + pB * d[TYPE_B]["screen"].mean()
        rows.append(("ENDPAC", "CT 1건당 암 발견(S1)", found / ct if ct > 0 else np.nan, np.nan, ""))
    return pd.DataFrame(rows, columns=["구분", "항목", "모델", "목표", "근거"])
