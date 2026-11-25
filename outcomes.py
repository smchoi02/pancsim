"""⑤ 결과: 생존, 비용, 효용, 할인 (설계서 7장). 시간 단위는 개월, 할인은 연 단위 이율을 연속 할인으로 환산."""
import numpy as np
from cohort import CTRL, U as _U


def _disc_int(a, b, p):
    """∫_a^b exp(-δ t) dt (t: 개월) / 12  -> '할인된 년' 단위."""
    d = np.log(1 + p["discount"]) / 12
    a = np.asarray(a, float); b = np.maximum(np.asarray(b, float), a)
    return (np.exp(-d * a) - np.exp(-d * b)) / d / 12


def _disc_at(t, p):
    return np.exp(-np.log(1 + p["discount"]) / 12 * np.asarray(t, float))


def _surv_L_months(U, p, hr=1.0):
    """국한(P0 포함) 생존 시간(개월). 완치 혼합 모형: S(t) = π + (1-π) exp(-λt).
    π = cure_L, λ는 5년 생존 S5 = exp(-5 x surv_L_rate_yr)을 맞추도록 계산 (π=0이면 기존 지수분포와 같음).
    hr > 1: 위험비 hr만큼 위험이 낮은 '작은 국한 종양' (비례위험: S_small = S^(1/hr) -> S(T) = U^hr).
    완치(U^hr <= π)면 inf: 췌장암으로 사망하지 않는다."""
    S5 = np.exp(-5 * p["surv_L_rate_yr"])
    pi = min(p.get("cure_L", 0.0), S5 - 0.01)
    lam = -np.log((S5 - pi) / (1 - pi)) / 5
    u = np.asarray(U, float) ** hr
    out = np.full(len(u), np.inf)
    m = u > pi
    out[m] = -np.log((u[m] - pi) / (1 - pi)) / lam * 12
    return out


def survival_months(stage, U, p):
    """진단 병기별 생존 시간(개월). P0·국한 = 완치 혼합 지수, 국소·원격 = 와이블. [2][20][49]"""
    out = np.empty(len(stage))
    m = stage <= 1
    out[m] = _surv_L_months(U[m], p)
    for s, key in ((2, "surv_R_weibull"), (3, "surv_D_weibull")):
        m = stage == s
        k, lam = p[key]
        out[m] = lam * (-np.log(U[m])) ** (1 / k) * 12
    return out


def within_stage(c, dx_t, det_state, p):
    """같은 단계에서 일찍 찾았을 때의 이득 (설계서 8.1절).

    g = (증상 진단 시각 - 발견 시각) / (증상 진단 시각 - 그 단계 진입 시각)  (0~1, 그 단계에서 남아 있던 비율)
    확률 w = 반영 비율 x g 로 '기준 생존'(국소 진행 → 국한 생존, 국한 → 작은 국한 종양 생존)을 따르고,
    아니면 원래 단계 생존을 따른다 (같은 생존 난수 사용). 기댓값은 원래 생존 + w x (기준 생존 - 원래 생존)과 같다.
    완치 모형에서는 기준 생존이 무한대(완치)일 수 있어 평균을 더하는 대신 확률로 나눈다. 원격 전이는 이득 없음.
    반환: (해당 여부, 확률 w, 기준 생존(개월))
    """
    n = len(dx_t)
    beta = p.get("within_stage_benefit", 0.0)
    s_eff = np.maximum(det_state, 1)                 # P0에서 찾은 경우도 국한 단계로 본다
    same = (beta > 0) & (det_state >= 0) & (s_eff == c["stage_c"]) & (dx_t < c["Tc"]) & (s_eff <= 2)
    w = np.zeros(n); ref = np.zeros(n)
    if not same.any():
        return same, w, ref
    Uv = c["U_surv"]; Tc = c["Tc"]
    entry = np.where(s_eff == 2, c["tR"], c["tL"])
    span = np.maximum(Tc - entry, 1e-9)
    g = np.clip((Tc - dx_t) / span, 0, 1)
    own = survival_months(c["stage_c"], Uv, p)
    ref_all = np.where(s_eff == 2, _surv_L_months(Uv, p), _surv_L_months(Uv, p, hr=p["size_hr_L"]))
    w = np.where(same, beta * g, 0.0)
    ref = np.where(same, np.maximum(ref_all, own), 0.0)
    return same, w, ref


def evaluate(c, scr, p, sname, kind):
    """사람별 할인 비용(만원)·QALY·생존연수와 임상 결과."""
    n = c["n"]
    horizon = (p["horizon_age"] - c["age"]) * 12
    bg = np.minimum(c["t_bg_death"], horizon)
    is_case = c["ctype"][0] != CTRL

    # --- 진단 시점과 병기 ---
    dx_t = np.full(n, np.inf); dx_s = np.full(n, -1)
    screen = np.zeros(n, bool)
    if is_case:
        det = scr["det_t"] < c["Tc"]
        realize = det & (c["U_theta"] < p["benefit_theta"])      # 이득이 실현되는 조기 발견
        dx_t = np.where(det, scr["det_t"], c["Tc"])
        dx_s = np.where(det, scr["det_state"], c["stage_c"])
        # 증상으로 진단됐을 때의 사망 시각 (선별이 없을 때와 같은 난수)
        death_clin = c["Tc"] + survival_months(c["stage_c"], c["U_surv"], p)
        # 선별로 일찍 찾았을 때: 발견 병기의 생존을 발견 시점부터 적용.
        # 단, 같은 병기에서 일찍 찾았다고 오히려 일찍 사망하는 일은 없도록(리드타임 착시 방지) 둘 중 늦은 시각.
        death_scr = np.maximum(dx_t + survival_months(dx_s, c["U_surv"], p), death_clin)
        # 같은 단계 안에서 일찍 찾은 경우의 이득 (설계서 8.1절): 확률 w(그 단계에서 남아 있던 기간의 비율 x 반영 비율)로
        # 기준 생존(국소 진행 → 국한 생존, 국한 → 작은 국한 종양 생존)을 따른다. 원격 전이는 이득 없음.
        same, w, ref = within_stage(c, dx_t, np.where(det, scr["det_state"], -1), p)
        take = same & (_U(c, "ws", 0) < w)
        death_scr = np.where(take, c["Tc"] + ref, death_scr)
        # 이득 미실현(benefit_theta): 일찍 찾았지만 사망 시각은 증상 진단 때와 같음
        ca_death = np.where(det & realize, death_scr, death_clin)
        alive_at_dx = dx_t < bg
        dx_t = np.where(alive_at_dx, dx_t, np.inf); dx_s = np.where(alive_at_dx, dx_s, -1)
        screen = det & alive_at_dx
        death = np.where(alive_at_dx, np.minimum(bg, ca_death), bg)
        died_cancer = alive_at_dx & (ca_death <= bg)
    else:
        death = bg
        died_cancer = np.zeros(n, bool)

    # --- QALY: 70세 전후 기준 효용, 진단 후 병기 배수 ---
    t70 = np.maximum((70 - c["age"]) * 12, 0)
    pre = np.minimum(dx_t, death)
    def base_q(a, b):
        return (p["u_base_lt70"] * _disc_int(a, np.minimum(b, np.maximum(t70, a)), p)
                + p["u_base_ge70"] * _disc_int(np.maximum(a, t70), np.maximum(b, t70), p))
    q = base_q(np.zeros(n), pre)
    if is_case:
        mult = np.array([p["u_mult"].get(int(s), 1.0) for s in np.clip(dx_s, 0, 3)])
        post = np.isfinite(dx_t)
        q = q + np.where(post, mult * base_q(np.where(post, dx_t, 0), np.where(post, death, 0)), 0)
    for t, m in scr["fp_times"]:
        q = q - m * p["disu_fp"][0] * p["disu_fp"][1] / 12 * _disc_at(t, p)
    for t, m in scr["ae_times"]:
        q = q - m * p["disu_eus_ae"][0] * p["disu_eus_ae"][1] / 12 * _disc_at(t, p)
    ly = _disc_int(np.zeros(n), death, p)

    # --- 비용 ---
    cost = np.zeros(n)
    # 패치 비용: 착용 회수 x 1회 추가 단가(CGM 사용자는 0) + 초음파 소자 1인당 1회
    if kind == "patch":
        cost += scr["n_wear"] * p["cost_patch_wear"] * _disc_at(6, p) + (scr["n_wear"] > 0) * p["cost_us_addon"]
    for t, m in scr["wp_times"]:
        cost += m * (p["cost_ct"] + p["cost_consult"]) * _disc_at(t, p)
    cost += scr["n_eus"] * p["cost_eus"] * _disc_at(6, p) + scr["n_ae"] * p["cost_eus_ae"] * _disc_at(6, p)
    if is_case:
        cost += cancer_costs(dx_t, death, died_cancer, p)

    return dict(cost=cost, qaly=q, ly=ly, screen=screen, dx_s=dx_s, dx_t=dx_t, death=death, died_cancer=died_cancer,
                fp=scr["n_fp"], ct=scr["n_ct"], eus=scr["n_eus"], wear=scr["n_wear"])


def cancer_costs(dx_t, death, died_cancer, p):
    """[19] Table 3 월 비용(달러 x 1100원 / 1만): 생존 구간에 따라 초기·지속·말기 단계 비용."""
    n = len(dx_t)
    cost = np.zeros(n)
    m = np.isfinite(dx_t)
    if not m.any():
        return cost
    s = (death - dx_t)[m]
    bucket = np.select([s < 12, s < 24, s < 36, s < 60], ["<1y", "1-2y", "2-3y", "3-5y"], ">5y")
    tab = p["cancer_cost_usd"]
    f = p["usd_krw"] / 1e4 * p.get("cancer_cost_mult", 1.0)
    ini = np.array([tab[b][0] for b in bucket]) * f
    con = np.array([tab[b][1] for b in bucket]) * f
    ter = np.array([tab[b][2] for b in bucket]) * f
    a = dx_t[m]; e = death[m]
    i_end = np.minimum(a + p["phase_init_m"], e)
    t_start = np.where(died_cancer[m], np.maximum(e - p["phase_term_m"], i_end), e)
    c_end = t_start
    if p["cont_cost_stop_after_5y"]:
        c_end = np.minimum(c_end, a + 60)
    c12 = 12  # _disc_int 은 '년' 단위이므로 월 비용 x 12
    cost_m = (ini * _disc_int(a, i_end, p) * c12 + con * _disc_int(i_end, np.maximum(c_end, i_end), p) * c12
              + ter * _disc_int(t_start, e, p) * c12)
    cost[m] = cost_m
    return cost
