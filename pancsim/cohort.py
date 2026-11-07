"""① 숨은 정답과 ② 몸의 실제 값 (설계서 3~4장).

층화 표본: 비암(대조군), A형, B형을 따로 만든 뒤 유병률로 가중해 합친다(설계서 8장, 계산 절약).
모든 난수는 사람 단위로 미리 뽑아 두고 모든 전략이 같은 값을 쓴다(공통 난수).
"""
import numpy as np
from scipy.stats import norm, truncnorm
from natural_history import simulate_paths
from lifetable import draw_death_age

CTRL, TYPE_A, TYPE_B = 0, 1, 2


def _tnorm(rng, mean, sd, lo, hi, n):
    a, b = (lo - mean) / sd, (hi - mean) / sd
    return truncnorm.rvs(a, b, loc=mean, scale=sd, size=n, random_state=rng)


def age_risk_slope(p, rng, n=200000):
    """나이가 많을수록 위험이 커지도록 P(암|나이) ∝ exp(beta*(나이-60))의 beta를 정한다.
    목표: 췌장암 환자 평균 나이 71세 [1]."""
    ages = _tnorm(rng, p["age_mean"], p["age_sd"], p["age_min"], p["age_max"], n)
    lo, hi = 0.0, 0.5
    for _ in range(50):
        b = (lo + hi) / 2
        w = np.exp(b * (ages - 60))
        if (w * ages).sum() / w.sum() < p["case_mean_age"]:
            lo = b
        else:
            hi = b
    return b


def make_cohort(ctype, n, p, nh, rng, beta_age):
    """ctype 집단 n명을 만든다. 반환: dict of arrays."""
    c = {"ctype": np.full(n, ctype), "n": n}
    # --- 나이·성별 (암 환자는 나이 가중 표본) ---
    if ctype == CTRL:
        c["age"] = _tnorm(rng, p["age_mean"], p["age_sd"], p["age_min"], p["age_max"], n)
    else:
        pool = _tnorm(rng, p["age_mean"], p["age_sd"], p["age_min"], p["age_max"], 8 * n)
        w = np.exp(beta_age * (pool - 60)); w /= w.sum()
        c["age"] = rng.choice(pool, size=n, p=w)
    c["male"] = rng.random(n) < p["male_frac"]
    c["t_bg_death"] = draw_death_age(c["age"], c["male"], rng.random(n), p)       # 개월

    # --- 자연경과 ---
    inf = np.full(n, np.inf)
    if ctype == CTRL:
        c.update(tL=inf.copy(), tR=inf.copy(), tD=inf.copy(), Tc=inf.copy(), stage_c=np.zeros(n, int), onset=inf.copy())
    elif ctype == TYPE_A:
        paths = simulate_paths(nh["nh"], 3 * n, rng, start_probs=nh["piA"])
        keep = np.where(paths["Tc"] <= p["screen_months"])[0][:n]                # 3년 안에 진단되는 사람이 '사례'
        assert len(keep) == n, "A형 표본 부족"
        for k in paths:
            c[k] = paths[k][keep]
        c["onset"] = np.zeros(n)
    else:
        got = {k: [] for k in ("tL", "tR", "tD", "Tc", "stage_c", "onset")}
        need = n
        while need > 0:
            m = 6 * need
            lo, hi = p.get("B_onset_range", (0, p["screen_months"]))
            onset = rng.uniform(lo, hi, m)
            paths = simulate_paths(nh["nh"], m, rng, onset=onset)
            ok = np.where((paths["Tc"] <= p["screen_months"]) & (paths["Tc"] > 0))[0][:need]
            for k in ("tL", "tR", "tD", "Tc", "stage_c"):
                got[k].append(paths[k][ok])
            got["onset"].append(onset[ok])
            need -= len(ok)
        for k in got:
            c[k] = np.concatenate(got[k])

    # --- 등록 전(검진 기록) 값: ENDPAC용 ---
    if ctype == TYPE_A:
        bb, bn, dw = p["bg_before_caseA"], p["bg_nod_caseA"], (-4.7, 5.9)       # [11] Table 3
    else:
        bb, bn, dw = p["bg_before_ctrl"], p["bg_nod_ctrl"], (1.3, 4.7)          # [11] Supp. Table 1
    c["bg_before"] = rng.normal(bb[0], bb[1], n)
    c["bg_nod"] = np.maximum(126, rng.normal(bn[0], bn[1], n))
    c["dw_pre_kg"] = rng.normal(dw[0], dw[1], n)
    c["endpac_ok"] = rng.random(n) < p["endpac_computable"]

    # --- 대조군형 배경 궤적 (모든 사람) ---
    z1, z2, z3 = rng.standard_normal((3, n))
    rc = p["rho_ctrl"]
    c["wt_rate"] = p["ctrl_wt_rate"][0] + p["ctrl_wt_rate"][1] * z1                       # %/년
    zd = rc * z1 + np.sqrt(1 - rc ** 2) * z2                                                # 체중과 같은 방향의 혈당 진행
    c["drift_mgdl_yr"] = p["adag_slope"] * (p["drift_hba1c_per_yr"][0] + p["drift_hba1c_per_yr"][1] * zd)
    k, r = p["sat_wt_ratio"], p["sat_wt_corr"]
    sd_e = k * p["ctrl_wt_rate"][1] * np.sqrt(1 / r ** 2 - 1)
    c["sat_rate"] = k * c["wt_rate"] + sd_e * z3                                           # %/년, 체중과 상관 r
    c["tx_drop"] = p["adag_slope"] * np.maximum(0, rng.normal(*p["tx_hba1c_drop"], n))      # mg/dL
    c["glp1"] = rng.random(n) < p["glp1_frac"]
    c["sat0"] = np.maximum(p["sat0_min"], rng.normal(*p["sat0"], n))

    # --- 암 신호 (A·B형) ---
    if ctype != CTRL:
        cv = p["glu_scale_cv"]; shape = 1 / cv ** 2
        u1, u2 = rng.standard_normal((2, n))
        rho = p["rho_case"]
        g_z = u1; f_z = rho * u1 + np.sqrt(1 - rho ** 2) * u2                               # 지수 방향 상관 rho
        c["glu_scale"] = np.asarray(_gamma_from_z(g_z, shape, 1 / shape)) * p["glu_excess_mult"] * (1.0 if ctype == TYPE_A else p["B_glu_mult"])
        c["fat_sig"] = rng.random(n) < p["fat_signal_frac"]
        c["fat_onset_m"] = rng.uniform(*p["fat_onset_range"], n)                            # 진단 전 개월
        lcv = p["fat_loss_cv"]; s = np.sqrt(np.log(1 + lcv ** 2))
        c["fat_loss_tot"] = np.minimum(0.6, p["fat_loss_total"] * np.exp(s * f_z - s ** 2 / 2))
    else:
        c["glu_scale"] = np.zeros(n); c["fat_sig"] = np.zeros(n, bool)
        c["fat_onset_m"] = np.zeros(n); c["fat_loss_tot"] = np.zeros(n)

    # --- 측정·순응·정밀검사용 공통 난수: 저장하지 않고 (시드, 월, 종류)로 매번 똑같이 다시 만든다 ---
    c["seed"] = int(rng.integers(0, 2**31 - 1))
    c["U_drop"] = rng.random(n)                       # 중단 시점 (지수분포 역변환)
    c["U_surv"] = rng.random(n); c["U_theta"] = rng.random(n)
    return c


_STREAMS = {"wear": 1, "cgm": 2, "us": 3, "reatt": 4, "scale_n": 5, "scale_z": 6,
            "acc": 7, "ct": 8, "ns": 9, "eus": 10, "ae": 11,
            "clin_g": 12, "clin_w": 13, "clin_f": 14}


def U(c, name, t, normal=False):
    """사람 x (월, 종류)별 공통 난수. 같은 코호트라면 어느 전략에서 불러도 같은 값."""
    g = np.random.default_rng([c["seed"], int(t), _STREAMS[name]])
    return g.standard_normal(c["n"]) if normal else g.random(c["n"])


def _gamma_from_z(z, shape, scale):
    from scipy.stats import gamma
    return gamma.ppf(norm.cdf(z), shape, scale=scale)


# ---------------- 몸의 실제 값 (시각 t 개월) ----------------
def _sah_excess(p, months_before):
    """[6] Figure 1 근사식의 (췌장암 - 대조군) 공복혈당 차이. x = 6 - 진단 전 개월/12."""
    a1, b1, c1 = p["sah_case"]; a0, b0, c0 = p["sah_ctrl"]
    x = np.clip(6 - months_before / 12.0, 1.32, 6.0)        # 1.32 = 차이 곡선의 최소점 (그 이전은 0에 가까움)
    return (a1 - a0) * x ** 2 + (b1 - b0) * x + (c1 - c0)


def true_glucose(c, t, p):
    g = c["bg_nod"] - c["tx_drop"] * (1 - np.exp(-t / p["tx_tau"])) + c["drift_mgdl_yr"] * t / 12
    g = g + np.where(c["glp1"], p["adag_slope"] * p["glp1_hba1c"] * (1 - np.exp(-t / 3.0)), 0.0)
    if c["ctype"][0] != CTRL:
        mb_t = np.maximum(c["Tc"] - t, 0); mb_0 = np.maximum(c["Tc"] - np.maximum(c["onset"], 0), 0)
        active = t >= c["onset"]
        ex = c["glu_scale"] * (_sah_excess(p, mb_t) - _sah_excess(p, mb_0))
        g = g + np.where(active, ex, 0.0)
    return g


def _fat_loss_frac(c, t):
    """진단 전 개월 기준 선형 감소: onset 시점 0 -> 진단 시 fat_loss_tot."""
    mb = c["Tc"] - t
    prog = np.clip((c["fat_onset_m"] - mb) / np.maximum(c["fat_onset_m"], 1e-9), 0, 1)
    return np.where(c["fat_sig"], c["fat_loss_tot"] * prog, 0.0)


def true_sat_and_weight(c, t, p):
    """복부 피하지방(mm)과 체중(기준 대비 %)."""
    yr = t / 12
    glp_w = np.where(c["glp1"], p["glp1_wt_pct"] * np.minimum(t / 6.0, 1.0), 0.0)        # 6개월에 걸쳐 감소 후 유지
    w_pct = 100 + c["wt_rate"] * yr + glp_w
    sat = c["sat0"] * (1 + (c["sat_rate"] * yr + p["sat_wt_ratio"] * glp_w) / 100)
    if c["ctype"][0] != CTRL:
        lf_t, lf_0 = _fat_loss_frac(c, t), _fat_loss_frac(c, 0.0)
        rel = (1 - lf_t) / (1 - lf_0)
        sat = sat * rel
        w_pct = w_pct * (1 - (1 - rel) / p["sat_wt_ratio"])
    return sat, w_pct
