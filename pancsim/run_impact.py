"""기대 효과 분석 실행 (본 분석).

사용법:
  python run_impact.py              # 전체 (약 15~30분, 컴퓨터에 따라 다름)
  python run_impact.py --fast       # 표본·반복을 줄인 빠른 확인 (약 3~4분)
결과: results_impact/기대효과.xlsx (그래프는 presentation/figs.py가 이 엑셀로 만든다)
"""
import argparse
import os
import time
import numpy as np
import pandas as pd
import params
import impact
from simulate import run_model
from validate import validation_table


def to_excel(path, sheets):
    with pd.ExcelWriter(path, engine="openpyxl") as xw:
        for name, d in sheets.items():
            d.to_excel(xw, sheet_name=name[:31], index=not isinstance(d.index, pd.RangeIndex))
        for ws in xw.book.worksheets:
            for col in ws.columns:
                w = max(len(str(c.value)) if c.value is not None else 0 for c in col)
                ws.column_dimensions[col[0].column_letter].width = min(max(10, w * 1.3), 50)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fast", action="store_true")
    ap.add_argument("--out", default="results_impact")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    big = (40_000, 4_000, 3_000) if a.fast else (200_000, 12_000, 8_000)
    sub = (20_000, 2_000, 1_500) if a.fast else (50_000, 5_000, 3_500)
    n_psa = 10 if a.fast else 60
    t0 = time.time(); log = lambda s: print(f"[{time.time() - t0:6.0f}s] {s}", flush=True)
    p = params.get(); nhd = params.load_nh(); nh = nhd["base"]
    sheets = {}

    log("1) 사용자 1만 명당 기대 효과")
    t = impact.impact_table(p, nh, sizes=big)
    sheets["1만명당_효과"] = t; print(t.round(2).to_string())

    log("2) 전국 연간 환산")
    nat = impact.national(t, p); sheets["전국_연간"] = nat; print(nat.round(1).to_string(index=False))

    log("3) 완벽한 기기 대비 실현 비율")
    h = impact.perfect_device(nh, p, sizes=sub)
    ours = t.loc["QALY (할인 4.5%)", "차이"]
    hr = pd.DataFrame([dict(항목="완벽한 기기의 QALY 증가 (1만 명당, 할인 4.5%)", 값=h["증분QALY_만명당"]),
                       dict(항목="우리 패치의 QALY 증가 (1만 명당, 할인 4.5%)", 값=ours),
                       dict(항목="실현 비율 (%)", 값=100 * ours / h["증분QALY_만명당"])])
    sheets["완벽한기기_대비"] = hr; print(hr.round(2).to_string(index=False))

    log("3b) 성능 지표, 병기 이동, CT 판독 수준별 결과와 놓친 횟수")
    dp = impact.device_pipeline(sizes=(100_000, 8_000, 6_000) if not a.fast else sub, log=log)
    sheets["성능지표"] = dp["성능"]; sheets["병기이동별_이득"] = dp["병기이동"]
    sheets["CT_판독수준"] = dp["판독수준"]; sheets["CT_놓침_단계별"] = dp["놓침_단계별"]
    for k in ("성능", "병기이동", "판독수준", "놓침_단계별"):
        print(dp[k].round(3).to_string(index=False))

    log("4) 검증")
    _, det, test = run_model(p, nh, *sub, seed=2026, strategies=["S0", "S1"], verbose=False)
    vt = validation_table(p, test, det); sheets["검증"] = vt

    log("5) 시나리오")
    st = impact.scenario_table(sizes=sub, log=log); sheets["시나리오"] = st
    print(st.round(2).to_string(index=False))

    log("5b) 혈당-피하지방 상관 민감도")
    ct = impact.correlation_table(sizes=sub, log=log); sheets["상관_민감도"] = ct
    print(ct.round(2).to_string(index=False))

    log(f"6) 확률 민감도 분석 {n_psa}회")
    raw, summ = impact.psa_impact(n_psa, sizes=sub, log=log)
    sheets["PSA_요약"] = summ; sheets["PSA_원자료"] = raw; print(summ.round(2).to_string())

    to_excel(f"{a.out}/기대효과.xlsx", sheets)
    log(f"완료 -> {a.out}/")


if __name__ == "__main__":
    main()
