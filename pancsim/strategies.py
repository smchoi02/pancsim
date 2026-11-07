"""③ 측정값과 ④ 판정·정밀검사 (설계서 5~7장).

전략은 세 가지만 둔다.
  S0   쓰지 않을 때 (현재 진료): 선별 없음, 증상이 생겨야 진단
  S9g  패치를 쓸 때 (본 분석): 착용마다 학습된 모형(그래디언트 부스팅)이 위험 확률을 계산하고,
       기준선을 넘으면 CT → 필요 시 내시경초음파 조직검사
  S1   ENDPAC → CT (모델 검증용): 등록 때 ENDPAC 3점 이상이면 정밀검사. 비교 전략이 아니라 검증 표(validate.py)에만 쓴다.
"""
import numpy as np
from cohort import U, true_glucose, true_sat_and_weight, CTRL
from natural_history import state_at

STRATEGIES = {
    "S0": dict(kind="none"),
    "S1": dict(kind="endpac"),
    "S9g": dict(kind="patch", sched="sched_S9", use_endpac=True),
}
LABELS = {"S0": "쓰지 않을 때 (현재 진료)", "S1": "ENDPAC → CT (검증용)", "S9g": "패치를 쓸 때"}


# ---------------- ENDPAC 점수 [11] Table 1 ----------------
def endpac_score(c):
    def cat_before(x):
        return np.select([x < 100, x < 110, x < 126], [1, 2, 3], 4)
    cat_nod = np.where(c["bg_nod"] > 160, 5, 4)
    A = cat_nod - cat_before(c["bg_before"])
    loss = -c["dw_pre_kg"]
    B = np.select([loss >= 6, loss >= 4, loss >= 2, loss > -2, loss > -4, loss > -6], [6, 4, 2, 0, -2, -4], -6)
    C = np.select([c["age"] < 60, c["age"] < 70], [-1, 0], 1)
    return A + B + C


def endpac_positive(c):
    return c["endpac_ok"] & (endpac_score(c) >= 3)


# ---------------- 측정 (5장) ----------------
def measure(c, t, p):
    """패치가 2주 동안 잰 평균 혈당(CGM)과 평균 복부 피하지방 두께(초음파)."""
    g = true_glucose(c, t, p)
    if p["cgm_err_cv"] > 0:
        g = g * (1 + p["cgm_err_cv"] / np.sqrt(14) * U(c, "cgm", t, True))           # 14일 야간 평균
    sat, _ = true_sat_and_weight(c, t, p)
    f = sat + p["reattach_sd"] * U(c, "reatt", t, True)
    if p["us_within_sd"] > 0:
        f = f + p["us_within_sd"] / np.sqrt(14 * 8) * U(c, "us", t, True)            # 하루 약 8회 x 14일 평균
    return g, f


class OnlineSlope:
    """회차가 쌓일 때마다 직선 기울기를 갱신 (최소제곱)."""
    def __init__(self, n):
        self.k = np.zeros(n); self.st = np.zeros(n); self.stt = np.zeros(n)
        self.sy = np.zeros(n); self.sty = np.zeros(n); self.first = np.full(n, np.nan)

    def add(self, mask, t_yr, y):
        m = mask & ~np.isnan(y)
        self.first = np.where(m & np.isnan(self.first), y, self.first)
        self.k[m] += 1; self.st[m] += t_yr; self.stt[m] += t_yr ** 2
        self.sy[m] += y[m]; self.sty[m] += t_yr * y[m]
        return m

    def slope(self):
        sxx = self.stt - self.st ** 2 / np.maximum(self.k, 1)
        b = (self.sty - self.st * self.sy / np.maximum(self.k, 1)) / np.where(sxx > 1e-12, sxx, np.nan)
        return np.where(self.k >= 2, b, np.nan)


# ---------------- 선별 진행 ----------------
def run_screening(c, p, sname, ref=None, collect=False):
    """한 집단 c에 전략 sname을 적용한다.
    collect=True: 판정 없이 정해진 일정으로 판정 특징만 모은다(모형 학습용).
    반환: dict(det_t, det_state, n_wear, n_ct, n_eus, n_ae, n_fp, fp_times, ae_times, wp_times)
      n_fp / fp_times: 암이 확인되지 않은 정밀검사 (종양이 없었으면 위양성, 있었으면 확진 단계의 위음성)
    """
    st = STRATEGIES[sname]; kind = st["kind"]; n = c["n"]
    is_case = c["ctype"][0] != CTRL
    out = dict(det_t=np.full(n, np.inf), det_state=np.full(n, -1), n_wear=np.zeros(n), n_ct=np.zeros(n),
               n_eus=np.zeros(n), n_ae=np.zeros(n), n_fp=np.zeros(n), fp_times=[], ae_times=[], wp_times=[])
    if kind == "none":
        return out
    end_t = np.minimum(c["Tc"], c["t_bg_death"])                        # 증상 진단 또는 사망 전까지만 선별
    adh = p["wear_adherence"]
    drop = p["annual_dropout"]
    t_drop = np.full(n, np.inf) if drop <= 0 else -np.log(c["U_drop"]) / (-np.log(1 - drop) / 12)

    # 정밀검사에서 암이 확인되지 않은 사람은 일정 기간(ct_cooldown_m개월) 다시 정밀검사로 보내지 않는다
    next_ok = np.zeros(n)

    def workup(mask, t):
        acc = mask & (U(c, "acc", t) < p["workup_accept"])
        out["n_ct"][acc] += 1
        out["wp_times"].append((t, acc.copy()))
        s = state_at(c, t) if is_case else np.full(n, -1)
        has_tumor = s >= 0
        ct_sens = np.array([p["ct_sens"].get(int(x), 0) for x in np.clip(s, 0, 3)])
        eus_sens = np.array([p["eus_sens"].get(int(x), 0) for x in np.clip(s, 0, 3)])
        ct_pos = acc & has_tumor & (U(c, "ct", t) < ct_sens)
        ns = acc & ~ct_pos & (U(c, "ns", t) < p["ct_nonspecific"])
        eus = ct_pos | ns
        out["n_eus"][eus] += 1
        ae = eus & (U(c, "ae", t) < p["eus_ae"]); out["n_ae"][ae] += 1
        out["ae_times"].append((t, ae.copy()))
        conf = eus & has_tumor & (U(c, "eus", t) < eus_sens)
        new = conf & np.isinf(out["det_t"])
        out["det_t"][new] = t; out["det_state"][new] = s[new]
        fp = mask & ~conf
        out["n_fp"][fp] += 1; out["fp_times"].append((t, fp.copy()))
        next_ok[fp] = t + p.get("ct_cooldown_m", 0)

    if kind == "endpac":                                                # S1: 등록 때 1회
        workup(endpac_positive(c) & (end_t > 0), 0)
        return out

    # ---- S9g: 패치 ----
    zE = np.zeros(n)
    if st.get("use_endpac"):
        zE = np.where(c["endpac_ok"], (endpac_score(c) - p["endpac_ctrl_mean"]) / p["endpac_ctrl_sd"], 0.0)
    sched = set(p[st["sched"]])
    extra = np.zeros((p["screen_months"] + 5, n), bool)                 # 주의 신호로 추가된 착용
    sG, sF = OnlineSlope(n), OnlineSlope(n)
    g_first = np.full(n, np.nan); t_first = np.full(n, np.nan)
    coll = []
    for t in range(p["screen_months"] + 1):
        base = t in sched
        sch = np.full(n, base) | extra[t]
        active = sch & (end_t > t) & (t_drop > t) & np.isinf(out["det_t"])
        wear = active & (U(c, "wear", t) < adh) if not collect else (np.full(n, base) & (end_t > t))
        if not wear.any():
            continue
        out["n_wear"][wear] += 1
        g, f = measure(c, t, p)
        ty = t / 12
        newf = wear & np.isnan(g_first)
        g_first[newf] = g[newf]; t_first[newf] = t
        sG.add(wear & (t >= p.get("glu_trend_from", 0)), ty, g)
        sF.add(wear, ty, f)
        bG = sG.slope()
        bF = sF.slope() / sF.first * 100                                # %/년
        ready = sF.k >= 1                                               # 첫 착용부터 판정 (수준값 + 나이 + 검진기록)
        X = ml_features(c, t, g, f, g_first, t_first, sF, bG, bF, zE)
        if collect:
            m = wear & ready
            if is_case:
                m = m & (state_at(c, t) >= 0)                          # 암 환자는 종양이 있는 동안의 착용만 학습
            coll.append(dict(X=X[m]))
            continue
        dec = wear & ready
        if not dec.any():
            continue
        prob = ml_predict(ref, X)
        pos = dec & (prob > ref["c"]) & (t >= next_ok)
        if pos.any():
            workup(pos, t)
        alert = dec & ~pos & (prob > ref["c_alert"])
        for dt in p["alert_extra"]:
            if t + dt <= p["screen_months"]:
                extra[t + dt] |= alert
    if collect:
        return coll
    return out


def build_reference(c_ctrl, c_case, p, sname):
    """학습용 코호트로 판정 모형과 기준선을 만든다 (설계서 6장). S0·S1은 필요 없음."""
    if STRATEGIES[sname]["kind"] != "patch":
        return None
    cc = run_screening(c_ctrl, p, sname, collect=True)
    ca = run_screening(c_case, p, sname, collect=True)
    return ml_train(cc, ca, p)


# ---------------- 판정 모형 ----------------
ML_FEATURES = ["나이", "검진ENDPAC(표준화)", "ENDPAC계산가능", "첫혈당", "현재혈당", "혈당변화", "혈당기울기",
               "혈당기울기있음", "피하지방변화%", "피하지방기울기%/년", "첫착용후개월", "피하지방측정횟수"]


def ml_features(c, t, g, f, g_first, t_first, sF, bG, bF, zE):
    """그 시점까지 패치가 잰 값 + 나이 + 검진기록으로 만든 판정 특징 12개."""
    fchg = (f / sF.first - 1) * 100
    return np.nan_to_num(np.column_stack([
        c["age"], zE, c["endpac_ok"].astype(float), g_first, g, g - g_first,
        np.nan_to_num(bG), (~np.isnan(bG)).astype(float), np.nan_to_num(fchg),
        np.nan_to_num(bF), t - np.nan_to_num(t_first), sF.k]))


def ml_train(cc, ca, p):
    """그래디언트 부스팅 학습 (암·비암 균형 가중). 기준선 = 대조군 확률의 target_spec 분위수, 주의 신호 = 90번째 분위수."""
    from sklearn.ensemble import HistGradientBoostingClassifier
    Xc = np.vstack([d["X"] for d in cc]); Xa = np.vstack([d["X"] for d in ca])
    X = np.vstack([Xc, Xa]); y = np.r_[np.zeros(len(Xc)), np.ones(len(Xa))]
    w = np.where(y == 1, len(Xc) / max(len(Xa), 1), 1.0)
    model = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.05, max_depth=3, l2_regularization=1.0,
                                           early_stopping=False, random_state=0)   # 재현성: 항상 200회, 같은 결과
    model.fit(X, y, sample_weight=w)
    pc = model.predict_proba(Xc)[:, 1]
    return dict(model=model, c=np.quantile(pc, p["target_spec"]), c_alert=np.quantile(pc, 0.90))


def ml_predict(ref, X):
    return ref["model"].predict_proba(X)[:, 1]
