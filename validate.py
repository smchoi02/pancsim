"""모델 검증 (설계서 9장): 모델이 만든 값이 실제 자료와 맞는지 확인한다.

1) 진단 시점 분포 (일반 진료)  vs  Chari 2025 [1] 0-4/4-12/12-24/24-36개월 = 30.5/31.3/19.5/18.7%
2) 진단 병기 분포 (일반 진료)   vs  국가암등록 [2] 국한 14.9 / 국소 35.2 / 원격 50.0% (병기 미상 제외)
3) 진단 후 5년 생존             vs  국가암등록 [2] 췌장암 5년 상대생존 약 16%
4) ENDPAC 성능 (S1)             vs  Sharma 2018 [11] 검증 코호트 민감도 78%, 특이도 85%(미국, 초록)
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
             ("ENDPAC", "특이도", spec, 0.72, "영국 외부검증 [28] (미국 원논문 0.85)"),
             ("ENDPAC", "양성예측도", ppv, np.nan, "")]

    # 5) 전략별 실제 정밀검사 결과 요약
    if "S1" in detail:
        d = detail["S1"]
        ct = sum(W * d[k]["ct"].mean() for k, W in ((CTRL, pC), (TYPE_A, pA), (TYPE_B, pB)))
        found = pA * d[TYPE_A]["screen"].mean() + pB * d[TYPE_B]["screen"].mean()
        rows.append(("ENDPAC", "CT 1건당 암 발견(S1)", found / ct if ct > 0 else np.nan, np.nan, ""))
    return pd.DataFrame(rows, columns=["구분", "항목", "모델", "목표", "근거"])


def realism_table(p, test, vt):
    """가상 코호트가 얼마나 현실적인지 세 단계로 정리한다.
    (1) 입력 재현: 논문 값을 넣어 만든 값이 그대로 나오는지 (프로그램 오류 확인)
    (2) 보정 목표: 자연경과 보정(ABC) 때 맞추도록 한 값 → 얼마나 맞췄는지
    (3) 독립 검증: 보정에 쓰지 않은 값 → 모델이 '맞힌' 값 (현실성의 근거)"""
    cA, cC = test[TYPE_A], test[CTRL]
    rows = [
        ("입력 재현", "당뇨 진단 1년 전 공복혈당 - 췌장암 (mg/dL)", cA["bg_before"].mean(), 110, "Sharma 2018 [11] Table 3"),
        ("입력 재현", "당뇨 진단 때 공복혈당 - 췌장암 (mg/dL)", cA["bg_nod"].mean(), 162, "Sharma 2018 [11] Table 3"),
        ("입력 재현", "당뇨 진단 1년 전 공복혈당 - 일반 당뇨 (mg/dL)", cC["bg_before"].mean(), 116, "Sharma 2018 [11] Supp. Table 1"),
        ("입력 재현", "당뇨 진단 때 공복혈당 - 일반 당뇨 (mg/dL)", cC["bg_nod"].mean(), 147, "Sharma 2018 [11] Supp. Table 1"),
        ("입력 재현", "당뇨 진단 전 1년 체중 변화 - 췌장암 (kg)", cA["dw_pre_kg"].mean(), -4.7, "Sharma 2018 [11] Table 3"),
        ("입력 재현", "당뇨 진단 전 1년 체중 변화 - 일반 당뇨 (kg)", cC["dw_pre_kg"].mean(), 1.3, "Sharma 2018 [11] Supp. Table 1"),
        ("입력 재현", "진단 전 피하지방이 줄어든 환자 비율", cA["fat_sig"].mean(), 0.71, "Babic 2025 [7]"),
        ("입력 재현", "진단까지 피하지방 감소율 (중앙값)", np.median(cA["fat_loss_tot"][cA["fat_sig"]]), 0.114, "Babic 2025 [7] Table 2"),
    ]
    stage = {"진단 시점": "보정 목표", "진단 병기(S0)": "보정 목표"}
    for _, r in vt.iterrows():
        if pd.isna(r["목표"]):
            continue
        st = stage.get(r["구분"], "독립 검증")
        if "국한" in r["항목"] and r["구분"] == "생존":
            st = "입력 재현"                                   # 국한 5년 생존은 입력값(λ를 이 값에 맞춤)
        rows.append((st, f'{r["구분"]} - {r["항목"]}', r["모델"], r["목표"], r["근거"]))
    d = pd.DataFrame(rows, columns=["단계", "항목", "가상 코호트", "실제 자료", "출처"])
    pct = ~d["항목"].str.contains(r"\(mg/dL\)|\(kg\)")
    d["차이"] = d["가상 코호트"] - d["실제 자료"]
    d["차이 단위"] = np.where(pct, "%p", np.where(d["항목"].str.contains("kg"), "kg", "mg/dL"))
    d.loc[pct, "차이"] *= 100
    d["비고"] = np.where(d["항목"].str.contains("당뇨 진단 때 공복혈당"), "126 미만은 당뇨가 아니므로 126으로 잘라 평균이 조금 높아짐", "")
    order = {"입력 재현": 0, "보정 목표": 1, "독립 검증": 2}
    d = d.sort_values("단계", key=lambda x: x.map(order), kind="stable").reset_index(drop=True)
    return d
