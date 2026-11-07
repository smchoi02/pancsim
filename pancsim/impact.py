"""기대 효과 분석 (본 분석).

질문: 50세 이상 신규 당뇨 환자가 패치를 설명서대로 쓰면, 쓰지 않을 때(현재 진료)보다 무엇이 얼마나 달라지나?
- 사용법(설명서): 당뇨 진단 후 0·3·6·12·18·24·30·36개월에 2주씩 착용(3년 8회). 정밀검사에서 암이 확인되지 않으면 12개월간 재검하지 않는다.
- 확진은 CT → 필요 시 내시경초음파 조직검사. CT 판독 수준(현재·PANDA·100%)에 따른 차이와 놓친 횟수는 device_pipeline()이 따로 계산한다.
  기본 분석은 설명서대로 사용(모든 회차 착용, 중단 없음,
  양성이면 모두 정밀검사)을 가정하고, 실제 사용(착용 80%, 연 27% 중단, 검사 수용 90%)은 시나리오로 본다.
- 판정: 앱 알고리즘(S9g)이 착용마다 위험 확률을 계산하고, 기준을 넘으면 CT 의뢰 → 필요 시 내시경초음파 조직검사. 알고리즘 입력: 패치 측정값(혈당·피하지방) + 나이 + 검진 기록(있으면).
- 비교: 같은 가상 환자가 패치를 쓰지 않은 경우(S0, 증상이 생겨야 진단).
결과는 '사용자 1만 명당'과 '전국 연간' 두 단위로 낸다(강의록 방식의 기대 효과).
"""
import numpy as np
import pandas as pd
import params
from simulate import run_model, build_cohorts, merge_cases
from cohort import CTRL, TYPE_A, TYPE_B, U
from strategies import STRATEGIES, run_screening, build_reference
from outcomes import evaluate
from natural_history import state_at

DEVICE = "S9g"          # 설명서대로 쓰는 패치 (최적화 판정 알고리즘)
BASE = "S0"             # 현재 진료

# 전국 환산: 50세 이상 신규 당뇨 연간 발생 수 (추정, 「기대효과 분석 정리」 5.5절)
NATIONAL_NOD_PER_YEAR = 150_000
NATIONAL_RANGE = (100_000, 250_000)


def _per10k(p, det):
    """전략별 '1만 명당' 합계 지표."""
    pC = 1 - p["prev_total_3y"]; pA = p["prev_total_3y"] - p["prev_B_3y"]; pB = p["prev_B_3y"]
    W = {CTRL: pC, TYPE_A: pA, TYPE_B: pB}
    out = {}
    for s, d in det.items():
        tot = lambda k: 1e4 * sum(W[ct] * np.mean(d[ct][k]) for ct in d)
        cases = lambda f: 1e4 * sum(W[ct] * np.mean(f(d[ct])) for ct in (TYPE_A, TYPE_B))
        out[s] = {
            "진단된 췌장암": cases(lambda x: x["dx_s"] >= 0),
            "증상 전 발견": cases(lambda x: x["screen"]),
            "국한(초기) 진단": cases(lambda x: np.isin(x["dx_s"], [0, 1])),
            "국소 진행 진단": cases(lambda x: x["dx_s"] == 2),
            "원격 전이(4기) 진단": cases(lambda x: x["dx_s"] == 3),
            "생존연수": tot("ly"), "QALY": tot("qaly"), "총비용(만원)": tot("cost"),
            "CT": tot("ct"), "내시경초음파": tot("eus"), "암 미확인 정밀검사": tot("fp"), "착용 횟수": tot("wear"),
        }
    return pd.DataFrame(out).T


def impact_table(p=None, nh=None, sizes=(100_000, 8_000, 6_000), seed=2026):
    """사용자 1만 명당 기대 효과. 생존연수는 할인 없이, QALY와 비용은 할인(4.5%) 있는 값과 없는 값을 함께."""
    p = p or params.get(); nh = nh or params.load_nh()["base"]
    rows = {}
    for disc, tag in ((1e-9, "할인 없음"), (p["discount"], "할인 적용")):
        q = dict(p, discount=disc)
        _, det, _ = run_model(q, nh, *sizes, seed=seed, strategies=[BASE, DEVICE], verbose=False)
        rows[tag] = _per10k(q, {k: det[k] for k in (BASE, DEVICE)})
    t = rows["할인 없음"].T.rename(columns={BASE: "쓰지 않을 때", DEVICE: "패치를 쓸 때"})
    t["차이"] = t["패치를 쓸 때"] - t["쓰지 않을 때"]
    d = rows["할인 적용"]
    extra = pd.DataFrame({"쓰지 않을 때": [np.nan, np.nan], "패치를 쓸 때": [np.nan, np.nan],
                          "차이": [d.loc[DEVICE, "QALY"] - d.loc[BASE, "QALY"], d.loc[DEVICE, "총비용(만원)"] - d.loc[BASE, "총비용(만원)"]]},
                         index=["QALY (할인 4.5%)", "총비용(만원, 할인 4.5%)"])
    return pd.concat([t, extra])


def national(t, p=None, n=NATIONAL_NOD_PER_YEAR, rng=NATIONAL_RANGE):
    """전국 연간 환산 (사용자 = 50세 이상 신규 당뇨 전원이라고 가정)."""
    p = p or params.get()
    keys = ["증상 전 발견", "원격 전이(4기) 진단", "국한(초기) 진단", "생존연수", "QALY", "CT", "암 미확인 정밀검사"]
    rows = []
    for k in keys:
        v = t.loc[k, "차이"] / 1e4
        rows.append(dict(항목=k, 기본=v * n, 하한=v * rng[0], 상한=v * rng[1]))
    v = t.loc["QALY", "차이"] / 1e4 * p["wtp"] / 1e4          # 억원
    rows.append(dict(항목="사회적 가치(억원, QALY×5,000만원, 기기비용 제외)", 기본=v * n, 하한=v * rng[0], 상한=v * rng[1]))
    return pd.DataFrame(rows)


# ---------------- 불확실성 ----------------
SCENARIOS = {
    "기본": {},
    "피하지방 감소 22% (Sah 2019)": dict(fat_loss_total=0.22),
    "피하지방 감소 6%": dict(fat_loss_total=0.06),
    "암 환자 혈당 상승 2배": dict(glu_excess_mult=2.0),
    "암 환자 혈당 상승 0.5배": dict(glu_excess_mult=0.5),
    "대조군 체중 변동 작음 (SD 2.5%/년)": dict(ctrl_wt_rate=(0.8, 2.5)),
    "재부착 오차 2mm": dict(reattach_sd=2.0),
    "실제 사용 (착용 80%, 연 27% 중단, 검사 수용 90%)": dict(wear_adherence=0.80, annual_dropout=0.27, workup_accept=0.90),
    "검진 기록 없음 (ENDPAC 0%)": dict(endpac_computable=0.0),
    "착용 1년만 (0·3·6·12개월)": dict(sched_S9=[0, 3, 6, 12]),
    "정밀검사 재검 제한 없음": dict(ct_cooldown_m=0),
    "당뇨 초과 사망 없음 (일반인 생명표)": dict(bg_mort_mult=(1.0, 1.0, 1.0, 1.0)),
    "당뇨 초과 사망 큼 (당화혈색소 7.0~7.8% 행)": dict(bg_mort_mult=(2.00, 1.55, 1.25, 1.03)),
    "같은 단계 이득 0%": dict(within_stage_benefit=0.0),
    "같은 단계 이득 100%": dict(within_stage_benefit=1.0),
    "이전 자연경과 보정 (등록 시 국소 진행 82%)": "oldnh",
}

# 혈당-피하지방 상관 민감도 (설계서 4.4절)
CORR_GRID = {
    "기본 (암 환자 0, 대조군 0.3)": {},
    "암 환자 -0.4": dict(rho_case=-0.4),
    "암 환자 +0.25": dict(rho_case=0.25),
    "암 환자 +0.5": dict(rho_case=0.5),
    "암 환자 +0.8": dict(rho_case=0.8),
    "대조군 0": dict(rho_ctrl=0.0),
    "대조군 0.6": dict(rho_ctrl=0.6),
    "암 환자 +0.8, 대조군 0": dict(rho_case=0.8, rho_ctrl=0.0),
}

# 확진(CT) 판독 수준 비교 (설계서 7.3절). PANDA: Cao 2023 Nat Med Fig. 2c 외부 검증 (T1 92.2%, 2기 96.5%, 4기 97%)
_ALL1 = {0: 1.0, 1: 1.0, 2: 1.0, 3: 1.0}
CT_LEVELS = {
    "현재 (문헌의 영상의학과 판독 수준)": {},
    "PANDA 판독 수준": dict(ct_sens={0: 0.10, 1: 0.922, 2: 0.965, 3: 0.97}),
    "CT 100% (P0 포함)": dict(ct_sens=_ALL1),
    "CT·조직검사 모두 100% (기기 순성능)": dict(ct_sens=_ALL1, eus_sens=_ALL1),
}


def _load_old_nh():
    import json, os
    with open(os.path.join(os.path.dirname(__file__), "calibrated_nh_old.json")) as f:
        return json.load(f)["base"]


def _grid_table(grid, sizes, seed, log):
    nhd = params.load_nh(); rows = []
    for name, ov in grid.items():
        nh = _load_old_nh() if ov == "oldnh" else nhd["base"]
        p = params.get(**({} if isinstance(ov, str) else ov), discount=1e-9)
        _, det, _ = run_model(p, nh, *sizes, seed=seed, strategies=[BASE, DEVICE], verbose=False)
        t = _per10k(p, {k: det[k] for k in (BASE, DEVICE)})
        dd = t.loc[DEVICE] - t.loc[BASE]
        rows.append(dict(시나리오=name, 증상전발견=dd["증상 전 발견"], 국한증가=dd["국한(초기) 진단"], 원격감소=-dd["원격 전이(4기) 진단"],
                         생존연수=dd["생존연수"], QALY=dd["QALY"], CT=dd["CT"], **{"암 미확인 정밀검사": dd["암 미확인 정밀검사"]}))
        if log:
            log(f"  {name}")
    return pd.DataFrame(rows)


def correlation_table(sizes=(50_000, 5_000, 3_500), seed=11, log=print):
    return _grid_table(CORR_GRID, sizes, seed, log)


def perfect_device(nh, p=None, sizes=(50_000, 5_000, 3_500), seed=11, every=3):
    """'완벽한 기기': 3개월마다 100% 착용, 종양이 있으면 100% 발견, 위양성 0.
    같은 자연경과에서 어떤 기기든 낼 수 있는 최대 효과. 우리 패치의 실현 비율(%)을 구하는 데만 쓴다."""
    p = p or params.get()
    test = build_cohorts(p, nh, *sizes, seed)
    W = {CTRL: 1 - p["prev_total_3y"], TYPE_A: p["prev_total_3y"] - p["prev_B_3y"], TYPE_B: p["prev_B_3y"]}
    res = {}
    for name, perfect in (("S0", False), ("완벽기기", True)):
        tot = dict(cost=0.0, qaly=0.0, early=0.0)
        for ct, c in test.items():
            n = c["n"]
            scr = dict(det_t=np.full(n, np.inf), det_state=np.full(n, -1), n_wear=np.zeros(n), n_ct=np.zeros(n),
                       n_eus=np.zeros(n), n_ae=np.zeros(n), n_fp=np.zeros(n), fp_times=[], ae_times=[], wp_times=[])
            if perfect and ct != CTRL:
                for t in range(0, p["screen_months"] + 1, every):
                    s = state_at(c, t)
                    new = (s >= 0) & np.isinf(scr["det_t"]) & (t < c["Tc"]) & (t < c["t_bg_death"])
                    scr["det_t"][new] = t; scr["det_state"][new] = s[new]
            ev = evaluate(c, scr, p, name, "none")
            tot["cost"] += W[ct] * ev["cost"].mean(); tot["qaly"] += W[ct] * ev["qaly"].mean()
            if ct != CTRL:
                tot["early"] += W[ct] * ev["screen"].mean() / p["prev_total_3y"]
        res[name] = tot
    dq = res["완벽기기"]["qaly"] - res["S0"]["qaly"]
    return dict(조기발견=res["완벽기기"]["early"], 증분QALY_만명당=1e4 * dq)


def device_pipeline(sizes=(100_000, 8_000, 6_000), seed=2026, log=print):
    """기본 설정의 성능 지표, 병기 이동별 이득, CT 판독 수준별 결과와 놓친 횟수를 한 번에 계산한다.

    학습·시험 코호트와 판정 모형은 한 번만 만들고, 판독 수준만 바꿔 같은 사람들에게 다시 적용한다.
    """
    p = params.get(discount=1e-9); nh = params.load_nh()["base"]
    pA = p["prev_total_3y"] - p["prev_B_3y"]; pB = p["prev_B_3y"]; pC = 1 - p["prev_total_3y"]
    W = {CTRL: pC, TYPE_A: pA, TYPE_B: pB}
    nC, nA, nB = sizes
    train = build_cohorts(p, nh, nC // 2, nA // 2, nB // 2, seed + 1)
    test = build_cohorts(p, nh, nC, nA, nB, seed)
    ref = build_reference(train[CTRL], merge_cases(train[TYPE_A], train[TYPE_B]), p, DEVICE)
    kind = STRATEGIES[DEVICE]["kind"]
    ev0 = {ct: evaluate(c, run_screening(c, p, BASE, None), p, BASE, "none") for ct, c in test.items()}
    nm = ["P0", "국한", "국소 진행", "원격 전이"]
    lv_rows, out = [], {}
    for lname, ov in CT_LEVELS.items():
        q = dict(p, **ov)
        scr = {ct: run_screening(c, q, DEVICE, ref) for ct, c in test.items()}
        ev = {ct: evaluate(c, scr[ct], q, DEVICE, kind) for ct, c in test.items()}
        tot = lambda f: 1e4 * sum(W[ct] * np.mean(f(ct)) for ct in test)
        cas = lambda f: 1e4 * sum(W[ct] * np.mean(f(ct)) for ct in (TYPE_A, TYPE_B))
        # 놓친 횟수: 종양이 있는 암 환자의 정밀검사 중 CT에서 안 보인 경우 / 조직검사에서 놓친 경우
        miss_ct = np.zeros(4); miss_eus = np.zeros(4); refer = np.zeros(4); never = 0.0
        pos_t = {}   # 종양이 있는 동안 알고리즘 양성(정밀검사 의뢰)이 한 번이라도 나온 사람 = 기기 민감도의 분자
        for ct in (TYPE_A, TYPE_B):
            c = test[ct]; r = scr[ct]; w = W[ct] / c["n"] * 1e4
            had = np.zeros(c["n"], bool)
            for t, acc in r["wp_times"]:
                st = state_at(c, t); tum = acc & (st >= 0)
                cs = np.array([q["ct_sens"].get(int(x), 0) for x in np.clip(st, 0, 3)])
                es = np.array([q["eus_sens"].get(int(x), 0) for x in np.clip(st, 0, 3)])
                ct_pos = tum & (U(c, "ct", t) < cs)
                ns = tum & ~ct_pos & (U(c, "ns", t) < q["ct_nonspecific"])
                conf = (ct_pos | ns) & (U(c, "eus", t) < es)
                for k in range(4):
                    m = st == k
                    refer[k] += w * (tum & m).sum()
                    miss_ct[k] += w * (tum & m & ~ct_pos & ~ns).sum()
                    miss_eus[k] += w * (tum & m & (ct_pos | ns) & ~conf).sum()
                had |= tum
            found = np.isfinite(r["det_t"]) & (r["det_t"] < c["Tc"])
            never += w * (had & ~found).sum()
            pos_t[ct] = had.copy()
        lv_rows.append({"판독 수준": lname,
                        "증상 전 발견": cas(lambda ct: ev[ct]["screen"]),
                        "국한 진단 증가": cas(lambda ct: np.isin(ev[ct]["dx_s"], [0, 1])) - cas(lambda ct: np.isin(ev0[ct]["dx_s"], [0, 1])),
                        "4기 진단 감소": cas(lambda ct: ev0[ct]["dx_s"] == 3) - cas(lambda ct: ev[ct]["dx_s"] == 3),
                        "생존연수": tot(lambda ct: ev[ct]["ly"] - ev0[ct]["ly"]),
                        "CT": tot(lambda ct: ev[ct]["ct"]),
                        "암 환자 CT 의뢰(종양 있음)": refer.sum(),
                        "CT에서 놓침": miss_ct.sum(), "조직검사에서 놓침": miss_eus.sum(),
                        "양성이었지만 끝내 증상 전 확진 못 함(명)": never})
        if lname.startswith("현재"):
            out["놓침_단계별"] = pd.DataFrame({"정밀검사 시점 종양 단계": nm, "암 환자 CT 의뢰": refer,
                                           "CT에서 놓침": miss_ct, "조직검사에서 놓침": miss_eus})
            # 성능 지표 (사람 기준, 3년)
            # 기기 성능(CT 이전): 암 환자는 종양이 있는 동안 양성이 나왔는지로 센다(B형의 종양 생기기 전 양성은 제외)
            pos = {CTRL: scr[CTRL]["n_ct"] > 0, TYPE_A: pos_t[TYPE_A], TYPE_B: pos_t[TYPE_B]}
            fnd = {ct: ev[ct]["screen"] for ct in (TYPE_A, TYPE_B)}
            wc = lambda f: (pA * f[TYPE_A].mean() + pB * f[TYPE_B].mean()) / (pA + pB)
            spec = 1 - pos[CTRL].mean()
            sens_sig = wc(pos); sens_fin = wc(fnd)
            ppv = sens_sig * (pA + pB) / (sens_sig * (pA + pB) + (1 - spec) * pC)
            out["성능"] = pd.DataFrame([
                dict(지표="[기기] 민감도 (종양이 있는 동안 알고리즘 양성이 한 번이라도 나온 암 환자 비율)", 전체=sens_sig, A형=pos[TYPE_A].mean(), B형=pos[TYPE_B].mean()),
                dict(지표="[전체 경로] 민감도 (CT·조직검사까지 거쳐 증상 전 확진된 비율)", 전체=sens_fin, A형=fnd[TYPE_A].mean(), B형=fnd[TYPE_B].mean()),
                dict(지표="[기기] 특이도 (암이 없는 사람 중 3년간 양성 없음)", 전체=spec, A형=np.nan, B형=np.nan),
                dict(지표="[기기] 양성예측도 (양성 판정자 중 그때 종양이 있던 사람)", 전체=ppv, A형=np.nan, B형=np.nan),
                dict(지표="[전체 경로] CT 1건당 암 확인", 전체=cas(lambda ct: ev[ct]["screen"]) / max(tot(lambda ct: ev[ct]["ct"]), 1e-9), A형=np.nan, B형=np.nan)])
            # 병기 이동별 이득
            rows = []
            for ct in (TYPE_A, TYPE_B):
                c = test[ct]; m = ev[ct]["screen"]; w = W[ct] / c["n"] * 1e4
                a = c["stage_c"][m]; b = np.maximum(ev[ct]["dx_s"][m], 1)
                gain = (ev[ct]["ly"] - ev0[ct]["ly"])[m]; lead = (c["Tc"] - scr[ct]["det_t"])[m]
                for x, y, g, l in zip(a, b, gain, lead):
                    rows.append((nm[x], nm[y], w, g, l))
            df = pd.DataFrame(rows, columns=["원래 단계", "발견 단계", "w", "g", "l"])
            out["병기이동"] = df.groupby(["원래 단계", "발견 단계"]).apply(
                lambda x: pd.Series({"만명당": x.w.sum(), "1명당 생존 증가(년)": np.average(x.g, weights=x.w),
                                     "평균 앞당김(개월)": np.average(x.l, weights=x.w)})).reset_index()
        if log:
            log(f"  판독 수준: {lname}")
    out["판독수준"] = pd.DataFrame(lv_rows)
    return out


def scenario_table(sizes=(50_000, 5_000, 3_500), seed=11, log=print):
    return _grid_table(SCENARIOS, sizes, seed, log)


def psa_impact(n_iter=60, sizes=(40_000, 4_000, 3_000), seed=7, log=print):
    """확률 민감도 분석: 입력값과 자연경과를 분포에서 뽑아 반복 → 1만 명당 효과의 분포."""
    import psa
    nhd = params.load_nh(); rng = np.random.default_rng(seed); rows = []
    for it in range(n_iter):
        p, nh, _ = psa.draw(rng, nhd)
        # 기기 성능 평가: 사용자 행동은 '설명서대로'로 고정하고 나머지 불확실성만 뽑는다
        p = dict(p, discount=1e-9, wear_adherence=1.0, annual_dropout=0.0, workup_accept=1.0)
        _, det, _ = run_model(p, nh, *sizes, seed=20_000 + it, strategies=[BASE, DEVICE], verbose=False)
        t = _per10k(p, {k: det[k] for k in (BASE, DEVICE)})
        dd = t.loc[DEVICE] - t.loc[BASE]
        rows.append(dict(반복=it, 암환자=t.loc[BASE, "진단된 췌장암"], 증상전발견=dd["증상 전 발견"],
                         원격감소=-dd["원격 전이(4기) 진단"], 생존연수=dd["생존연수"], QALY=dd["QALY"], CT=dd["CT"]))
        if log and (it + 1) % 10 == 0:
            log(f"  PSA {it + 1}/{n_iter}")
    raw = pd.DataFrame(rows)
    summ = raw.drop(columns="반복").describe(percentiles=[0.025, 0.5, 0.975]).T[["mean", "2.5%", "50%", "97.5%"]]
    summ.columns = ["평균", "2.5%", "중앙값", "97.5%"]
    return raw, summ
