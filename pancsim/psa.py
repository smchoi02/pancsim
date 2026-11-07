"""확률 민감도 분석 PSA (설계서 10.2절).

반복마다 불확실한 입력값을 분포에서 한 세트 뽑아 전체 모델을 다시 돌린다.
- 자연경과: ABC 보정에서 받아들여진 파라미터 세트(calibrated_nh.json 'accepted') 중 하나를 고른다.
  -> 보정 불확실성이 그대로 결과에 전달된다.
- 나머지: 아래 SPEC의 분포. 근거가 약한(D등급) 값일수록 범위를 넓게 잡았다.
본 분석은 impact.psa_impact()가 draw()로 입력값을 뽑아 쓴다. 사용자 행동(착용률·중단·검사 수용)은 거기서 '설명서대로'로 고정한다.
"""
import numpy as np
import params


def _beta(rng, mean, sd):
    k = mean * (1 - mean) / sd ** 2 - 1
    return rng.beta(mean * k, (1 - mean) * k)


def _gamma_mult(rng, cv):
    s = 1 / cv ** 2
    return rng.gamma(s, 1 / s)


# (이름, 표본 함수, 설명/근거)
SPEC = [
    ("prev_total_3y", lambda r: _beta(r, 0.0062, 0.0010), "베타(평균 0.62%, SD 0.10%p) [1]"),
    ("prev_B_frac", lambda r: r.uniform(0.25, 0.55), "B형 비율 U(0.25,0.55) [4] 범위 0.0015~0.0035 / 0.0062"),
    ("fat_loss_total", lambda r: r.triangular(0.06, 0.114, 0.22), "삼각(0.06, 0.114, 0.22) [7] Table 2 ~ [6] Fig 4A"),
    ("fat_signal_frac", lambda r: _beta(r, 0.71, 0.05), "베타(0.71, SD 0.05) [7]"),
    ("ctrl_wt_sd", lambda r: r.uniform(3.0, 5.5), "대조군 체중 변화율 SD U(3.0,5.5)%/년 [9][11]"),
    ("glu_excess_mult", lambda r: np.exp(r.uniform(np.log(0.5), np.log(2.0))), "로그균등(0.5~2) D"),
    ("B_glu_mult", lambda r: r.uniform(0.0, 1.0), "U(0,1) D"),
    ("rho_case", lambda r: r.uniform(-0.4, 0.8), "U(-0.4,0.8) 설계서 4.4절 D"),
    ("within_stage_benefit", lambda r: r.uniform(0.0, 1.0), "같은 단계 이득 반영 비율 U(0,1) D"),
    ("size_hr_L", lambda r: r.uniform(1.35, 1.91), "2cm 초과 사망 위험비 U(1.35,1.91) [44] 95% CI"),
    ("bg_mort_mix", lambda r: r.uniform(0.0, 1.0), "당뇨 초과 사망: Tancredi 2015 Table 3의 6.9% 이하 행과 7.0~7.8% 행 사이 U(0,1)"),
    ("tx_drop_mean", lambda r: r.uniform(0.6, 1.4), "치료 HbA1c 감소 평균 U(0.6,1.4)%p D"),
    ("glp1_frac", lambda r: r.uniform(0.0, 0.2), "U(0,0.2) D"),
    ("endpac_computable", lambda r: r.uniform(0.163, 0.759), "U(0.163,0.759) [12][28]"),
    ("wear_adherence", lambda r: _beta(r, 0.80, 0.05), "베타(0.80, SD 0.05) [30]"),
    ("annual_dropout", lambda r: _beta(r, 0.27, 0.05), "베타(0.27, SD 0.05) [30]"),
    ("workup_accept", lambda r: r.uniform(0.8, 1.0), "U(0.8,1.0) D"),
    ("ct_sens_L", lambda r: _beta(r, 0.50, 0.10), "베타(0.50, SD 0.10) [24]"),
    ("eus_sens_L", lambda r: _beta(r, 0.944, 0.02), "베타(0.944, SD 0.02) [24]"),
    ("benefit_theta", lambda r: r.uniform(0.5, 1.0), "U(0.5,1.0) D"),
    ("surv_mult", lambda r: _gamma_mult(r, 0.10), "생존 시간 배수 감마(CV 0.10) [2][20]"),
    ("cancer_cost_mult", lambda r: _gamma_mult(r, 0.20), "감마(CV 0.20) [19]"),
    ("proc_cost_mult", lambda r: _gamma_mult(r, 0.20), "CT·내시경초음파 비용 배수 감마(CV 0.20) [29]"),
    ("device_cost_mult", lambda r: _gamma_mult(r, 0.25), "패치 비용 배수 감마(CV 0.25) D"),
    ("u_base_lt70", lambda r: _beta(r, 0.81, 0.02), "베타(0.81, SD 0.02) [18]"),
    ("u_mult_D", lambda r: _beta(r, 0.884, 0.03), "베타(0.884, SD 0.03) [20]"),
]


def draw(rng, nhd):
    """입력값 한 세트를 뽑아 (p, nh, 뽑은 값 dict) 를 돌려준다."""
    v = {name: float(f(rng)) for name, f, _ in SPEC}
    b = params.BASE
    p = params.get(
        prev_total_3y=v["prev_total_3y"], prev_B_3y=v["prev_total_3y"] * v["prev_B_frac"],
        fat_loss_total=v["fat_loss_total"], fat_signal_frac=v["fat_signal_frac"],
        ctrl_wt_rate=(b["ctrl_wt_rate"][0], v["ctrl_wt_sd"]),
        glu_excess_mult=v["glu_excess_mult"], B_glu_mult=v["B_glu_mult"], rho_case=v["rho_case"],
        tx_hba1c_drop=(v["tx_drop_mean"], b["tx_hba1c_drop"][1]), glp1_frac=v["glp1_frac"],
        endpac_computable=v["endpac_computable"], wear_adherence=v["wear_adherence"],
        annual_dropout=v["annual_dropout"], workup_accept=v["workup_accept"],
        ct_sens={**b["ct_sens"], 1: v["ct_sens_L"]}, eus_sens={**b["eus_sens"], 1: v["eus_sens_L"]},
        benefit_theta=v["benefit_theta"],
        surv_L_rate_yr=b["surv_L_rate_yr"] / v["surv_mult"],
        surv_R_weibull=(b["surv_R_weibull"][0], b["surv_R_weibull"][1] * v["surv_mult"]),
        surv_D_weibull=(b["surv_D_weibull"][0], b["surv_D_weibull"][1] * v["surv_mult"]),
        cancer_cost_mult=v["cancer_cost_mult"],
        within_stage_benefit=v["within_stage_benefit"], size_hr_L=v["size_hr_L"],
        bg_mort_mult=tuple(lo + v["bg_mort_mix"] * (hi - lo) for lo, hi in zip((1.92, 1.40, 1.11, 0.95), (2.00, 1.55, 1.25, 1.03))),
        cost_ct=b["cost_ct"] * v["proc_cost_mult"], cost_eus=b["cost_eus"] * v["proc_cost_mult"],
        cost_patch_wear=b["cost_patch_wear"] * v["device_cost_mult"],
        u_base_lt70=v["u_base_lt70"], u_base_ge70=v["u_base_lt70"] - 0.04,
        u_mult={**b["u_mult"], 3: v["u_mult_D"]},
    )
    acc = nhd["accepted"]
    nh = acc[int(rng.integers(len(acc)))]
    return p, nh, v
