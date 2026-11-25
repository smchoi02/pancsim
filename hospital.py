"""병원 관점: 췌장암 진료에 쓰이는 입원일수(병상 일수). 비용·QALY와 별개로 보고한다.

진단된 환자 1명의 입원일수(할인 없음, 진단부터 사망까지):
  초기   절제하면 수술 재원일수, 아니면 진단 첫 달 입원일수 (절제 비율은 진단 병기별 기댓값)
  추적기 진단 2개월째부터 사망 3개월 전까지, 최대 24개월, 월평균 입원일수
  말기   췌장암으로 사망한 경우에만, 사망 전 3개월 입원일수 (생존이 3개월보다 짧으면 비례해서 줄임)
"""
import numpy as np


def bed_days(dx_s, dx_t, death, died_cancer, p):
    """사람별 기대 입원일수와 그 구성(수술, 초기 비수술, 추적기, 말기)."""
    n = len(dx_s)
    out = {k: np.zeros(n) for k in ("수술", "초기 입원(비수술)", "추적기", "말기")}
    m = np.isfinite(dx_t) & (dx_s >= 0)
    if not m.any():
        return out
    s = np.clip(dx_s[m], 0, 3)
    rf = np.array([p["resect_frac"][int(x)] for x in s])
    surv = np.maximum(death[m] - dx_t[m], 0)                       # 개월
    out["수술"][m] = rf * p["los_surgery"]
    out["초기 입원(비수술)"][m] = (1 - rf) * p["los_first_month"] * np.minimum(surv, 1)
    end = np.where(died_cancer[m], surv - 3, surv)
    months = np.clip(np.minimum(end, p["los_followup_months"]) - 1, 0, None)
    out["추적기"][m] = months * p["los_per_month"]
    out["말기"][m] = np.where(died_cancer[m], p["los_terminal"] * np.minimum(surv / 3, 1), 0)
    return out
