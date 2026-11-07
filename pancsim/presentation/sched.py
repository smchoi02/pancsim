"""착용 일정 × 재검 제한 비교 (현재 자연경과·현재 모델). 결과: presentation/sched.csv (약 5분)

실행 (저장소 최상위에서):  python presentation/sched.py
"""
import os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import pandas as pd, params, impact
from simulate import run_model
nh = params.load_nh()["base"]
Y1 = [0, 3, 6, 12]; Y3 = [0, 3, 6, 12, 18, 24, 30, 36]; Q3 = list(range(0, 37, 3))
runs = [("1년 4회", Y1, 0), ("1년 4회", Y1, 12), ("3년 8회 (기본)", Y3, 12), ("3년 13회 (3개월마다)", Q3, 12), ("3년 8회", Y3, 0)]
rows = []
for name, sch, cd in runs:
    p = params.get(discount=1e-9, sched_S9=sch, ct_cooldown_m=cd)
    _, det, _ = run_model(p, nh, 100_000, 8_000, 6_000, seed=2026, strategies=["S0", "S9g"], verbose=False)
    t = impact._per10k(p, {k: det[k] for k in ("S0", "S9g")}); d = t.loc["S9g"] - t.loc["S0"]
    rows.append({"착용 일정": name, "재검 제한": "12개월" if cd else "없음", "증상 전 발견": d["증상 전 발견"],
                 "국한 진단 증가": d["국한(초기) 진단"], "4기 진단 감소": -d["원격 전이(4기) 진단"], "생존 기간 증가(년)": d["생존연수"],
                 "CT": d["CT"], "CT 1건당 암 확인(%)": 100 * d["증상 전 발견"] / d["CT"], "1인당 착용 횟수": t.loc["S9g", "착용 횟수"] / 1e4,
                 "1인당 추가 비용(만원)": d["총비용(만원)"] / 1e4})
    print(rows[-1], flush=True)
pd.DataFrame(rows).to_csv(os.path.join(HERE, "sched.csv"), index=False)
