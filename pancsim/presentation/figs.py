"""발표용 그래프: results_impact/기대효과.xlsx + 착용 일정 비교(presentation/sched.csv)로 비교별 PNG를 만든다.

실행 (저장소 최상위에서):  python presentation/figs.py
인자를 주면 그 경로를 쓴다:  python presentation/figs.py <결과 엑셀> <그림 폴더> <sched.csv>
"""
import sys
import os
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm

args = sys.argv[1:] + [None] * 3
XLSX = args[0] or os.path.join(ROOT, "results_impact", "기대효과.xlsx")
OUT = args[1] or os.path.join(HERE, "figures")
SCHED = args[2] or os.path.join(HERE, "sched.csv")
os.makedirs(OUT, exist_ok=True)

for f in fm.findSystemFonts():
    if "NotoSansCJK" in f:
        fm.fontManager.addfont(f)
_have = {f.name for f in fm.fontManager.ttflist}
KFONT = next((n for n in ("Noto Sans CJK KR", "Malgun Gothic", "AppleGothic", "NanumGothic") if n in _have), "sans-serif")
plt.rcParams.update({
    "font.family": KFONT, "font.size": 12, "axes.unicode_minus": False,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.edgecolor": "#8a8984", "axes.labelcolor": "#52514e",
    "xtick.color": "#52514e", "ytick.color": "#52514e",
    "figure.facecolor": "#ffffff", "axes.facecolor": "#ffffff",
})
INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#8a8984", "#e6e5e0"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"   # 검증한 범주 색 1~3번 (dataviz 기본 팔레트)
GRAY = "#b9b7ae"                                        # '쓰지 않을 때'(비교 기준)
NOTE = "가상 코호트 시뮬레이션 · 50세 이상 신규 당뇨 · 설명서대로 사용 · 3년"

x = pd.read_excel(XLSX, sheet_name=None)
made = []


def fig(w=10, h=5.6):
    f, ax = plt.subplots(figsize=(w, h), dpi=200)
    return f, ax


def finish(f, name, title=None, sub=None, note=None, top=None):
    """그림에는 그래프만 남긴다 (제목·설명·출처 문구 없음). title/sub/note는 호출 호환용으로 받고 쓰지 않는다."""
    f.subplots_adjust(top=0.95, bottom=0.12)
    f.savefig(f"{OUT}/{name}.png", dpi=200)
    plt.close(f)
    made.append(name)


def hgrid(ax):
    ax.yaxis.grid(True, color=GRID, lw=0.8); ax.set_axisbelow(True)


def vgrid(ax):
    ax.xaxis.grid(True, color=GRID, lw=0.8); ax.set_axisbelow(True)


def bar_labels(ax, bars, fmt, inside=False, color=INK, dy=0.0, fs=11):
    for b in bars:
        v = b.get_height()
        ax.text(b.get_x() + b.get_width() / 2, v + dy, fmt(v), ha="center", va="bottom", fontsize=fs, color=color)


def hbar_labels(ax, bars, fmt, dx, fs=11):
    for b in bars:
        v = b.get_width()
        ax.text(v + dx if v >= 0 else v - dx, b.get_y() + b.get_height() / 2, fmt(v),
                ha="left" if v >= 0 else "right", va="center", fontsize=fs, color=INK)


def table_img(name, header, rows, colw, h=None):
    """표만 그린 그림 (제목·설명 없음)."""
    n = len(rows) + 1
    f = plt.figure(figsize=(10, h or 0.42 * n + 0.3), dpi=200)
    ax = f.add_axes([0.01, 0.01, 0.98, 0.98]); ax.axis("off")
    tb = ax.table(cellText=rows, colLabels=header, colWidths=colw, loc="upper center", cellLoc="right", colLoc="center")
    tb.auto_set_font_size(False); tb.set_fontsize(11.5)
    for (r, c), cell in tb.get_celld().items():
        cell.set_edgecolor("#c9c8c2"); cell.set_height(0.98 / n)
        if r == 0:
            cell.set_facecolor(BLUE); cell.get_text().set_color("white"); cell.get_text().set_fontweight("bold")
        else:
            cell.set_facecolor("white")
            if c == 0:
                cell.get_text().set_ha("left"); cell._loc = "left"
    f.savefig(f"{OUT}/{name}.png", dpi=200)
    plt.close(f)
    made.append(name)


# ======================================================================
# 01 기본 비교: 진단 병기 분포
e = x["1만명당_효과"].set_index(x["1만명당_효과"].columns[0])
stages = ["국한(초기) 진단", "국소 진행 진단", "원격 전이(4기) 진단"]
lab = ["국한 (초기)", "국소 진행", "원격 전이 (4기)"]
f, ax = fig()
xi = np.arange(3); w = 0.36
b0 = ax.bar(xi - w / 2 - 0.01, e.loc[stages, "쓰지 않을 때"], w, color=GRAY, label="쓰지 않을 때 (현재 진료)")
b1 = ax.bar(xi + w / 2 + 0.01, e.loc[stages, "패치를 쓸 때"], w, color=BLUE, label="패치를 쓸 때")
bar_labels(ax, b0, lambda v: f"{v:.1f}", dy=0.4); bar_labels(ax, b1, lambda v: f"{v:.1f}", dy=0.4)
for i, s in enumerate(stages):
    d = e.loc[s, "차이"]
    ax.text(i, max(e.loc[s, "쓰지 않을 때"], e.loc[s, "패치를 쓸 때"]) + 3.2, f"{d:+.1f}명".replace("-", "−"), ha="center", fontsize=12.5,
            fontweight="bold", color=INK)
ax.set_xticks(xi, lab, fontsize=12.5); ax.set_ylabel("진단되는 췌장암 환자 (명)"); ax.set_ylim(0, 38); hgrid(ax)
ax.legend(frameon=False, loc="upper left", fontsize=11.5)
finish(f, "01_병기분포_패치사용여부", "패치를 쓰면 4기 진단이 줄고 초기 진단이 늘어난다",
       "사용자 1만 명당 3년간 진단 병기별 췌장암 환자 수 (전체 환자 수는 약 60명으로 같음)")

# 02 핵심 효과 숫자
wear = e.loc["착용 횟수", "패치를 쓸 때"] / 1e4
D = lambda k: e.loc[k, "차이"]
rows02 = [
    ["진단되는 췌장암 환자 (명)", f"{e.loc['진단된 췌장암', '쓰지 않을 때']:.1f}", f"{e.loc['진단된 췌장암', '패치를 쓸 때']:.1f}", f"{D('진단된 췌장암'):+.1f}"],
    ["증상 전 발견 (명)", "0.0", f"{e.loc['증상 전 발견', '패치를 쓸 때']:.1f}", f"+{D('증상 전 발견'):.1f}"],
    ["국한(초기) 진단 (명)", f"{e.loc['국한(초기) 진단', '쓰지 않을 때']:.1f}", f"{e.loc['국한(초기) 진단', '패치를 쓸 때']:.1f}", f"+{D('국한(초기) 진단'):.1f}"],
    ["국소 진행 진단 (명)", f"{e.loc['국소 진행 진단', '쓰지 않을 때']:.1f}", f"{e.loc['국소 진행 진단', '패치를 쓸 때']:.1f}", f"+{D('국소 진행 진단'):.1f}"],
    ["원격 전이(4기) 진단 (명)", f"{e.loc['원격 전이(4기) 진단', '쓰지 않을 때']:.1f}", f"{e.loc['원격 전이(4기) 진단', '패치를 쓸 때']:.1f}", f"−{-D('원격 전이(4기) 진단'):.1f}"],
    ["생존 기간 증가 (년)", "", "", f"+{D('생존연수'):.1f}"],
    ["QALY 증가 (할인 없음 / 4.5%)", "", "", f"+{D('QALY'):.1f} / +{e.loc['QALY (할인 4.5%)', '차이']:.1f}"],
    ["추가 CT (건)", "0", f"{e.loc['CT', '패치를 쓸 때']:,.0f}", f"+{D('CT'):,.0f}"],
    ["추가 내시경초음파 조직검사 (건)", "0", f"{e.loc['내시경초음파', '패치를 쓸 때']:,.0f}", f"+{D('내시경초음파'):,.0f}"],
    ["암이 확인되지 않은 정밀검사 (회)", "0", f"{e.loc['암 미확인 정밀검사', '패치를 쓸 때']:,.0f}", f"+{D('암 미확인 정밀검사'):,.0f}"],
    ["CT 1건당 암 확인 (%)", "", "", f"{100 * D('증상 전 발견') / D('CT'):.1f}"],
    ["1인당 착용 횟수 (회)", "0", f"{wear:.1f}", f"+{wear:.1f}"],
    ["1인당 추가 비용 (만 원)", "", "", f"+{D('총비용(만원)') / 1e4:.1f}"],
]
table_img("02_핵심효과_1만명당", ["항목 (사용자 1만 명당, 3년)", "쓰지 않을 때", "패치를 쓸 때", "차이"], rows02, [0.46, 0.16, 0.16, 0.22])

# 03 성능 지표: 왼쪽 민감도(기기 / 전체 경로), 오른쪽 기기 특이도·양성예측도·CT 1건당 암 확인
pf = x["성능지표"]
f, (ax, ax2) = plt.subplots(1, 2, figsize=(10, 5.6), dpi=200, gridspec_kw=dict(width_ratios=[2.2, 1]))
grp = ["전체", "A형", "B형"]; glab = ["전체", "A형", "B형"]
sig = np.array([100 * float(pf.iloc[0][g]) for g in grp]); fin = np.array([100 * float(pf.iloc[1][g]) for g in grp])
xi = np.arange(3)
b0 = ax.bar(xi - w / 2 - 0.01, sig, w, color=BLUE, label="기기 민감도")
b1 = ax.bar(xi + w / 2 + 0.01, fin, w, color=ORANGE, label="전체 경로 민감도")
bar_labels(ax, b0, lambda v: f"{v:.1f}%", dy=0.8); bar_labels(ax, b1, lambda v: f"{v:.1f}%", dy=0.8)
ax.set_xticks(xi, glab, fontsize=12); ax.set_ylabel("민감도 (%)"); ax.set_ylim(0, 65); hgrid(ax)
ax.legend(frameon=False, loc="upper right", fontsize=11)
v2 = [100 * float(pf.iloc[2]["전체"]), 100 * float(pf.iloc[3]["전체"]), 100 * float(pf.iloc[4]["전체"])]
bb = ax2.bar(np.arange(3), v2, 0.6, color=[BLUE, BLUE, ORANGE])
bar_labels(ax2, bb, lambda v: f"{v:.1f}%", dy=1.2)
ax2.set_xticks(np.arange(3), ["기기\n특이도", "기기\n양성예측도", "CT 1건당\n암 확인"], fontsize=11); ax2.set_ylabel("%"); ax2.set_ylim(0, 105); hgrid(ax2)
f.subplots_adjust(wspace=0.3, left=0.08, right=0.98)
finish(f, "03_민감도_특이도")

# 04 병기 이동별 이득
sh = x["병기이동별_이득"].copy()
order = [("원격 전이", "국한"), ("원격 전이", "국소 진행"), ("국소 진행", "국한"), ("국소 진행", "국소 진행"), ("국한", "국한"), ("원격 전이", "원격 전이")]
sh = sh.set_index(["원래 단계", "발견 단계"]).loc[order].reset_index()
labs = [f"{a} → {b}" + (" (같은 단계)" if a == b else "") for a, b in order]
f, ax = fig()
yi = np.arange(len(sh))[::-1]
cols = [BLUE if a != b else ORANGE for a, b in order]
bars = ax.barh(yi, sh["1명당 생존 증가(년)"], 0.6, color=cols)
for y, (_, r) in zip(yi, sh.iterrows()):
    ax.text(r["1명당 생존 증가(년)"] + 0.05, y, f"{r['1명당 생존 증가(년)']:.1f}년   ({r['만명당']:.1f}명, 평균 {r['평균 앞당김(개월)']:.0f}개월 일찍)",
            va="center", fontsize=11, color=INK)
ax.set_yticks(yi, labs, fontsize=12); ax.set_xlabel("일찍 찾은 환자 1명당 늘어나는 생존 (년)"); ax.set_xlim(0, 5.2); vgrid(ax)
from matplotlib.patches import Patch
ax.legend(handles=[Patch(color=BLUE, label="더 이른 단계로 이동"), Patch(color=ORANGE, label="같은 단계에서 일찍 발견")],
          frameon=False, loc="lower right", fontsize=11)
f.subplots_adjust(left=0.27)
finish(f, "04_병기이동별_이득", "4기 대신 국한 단계에서 찾으면 1명당 약 3.3년을 더 산다",
       "증상으로 진단됐을 단계 → 패치로 찾은 단계 · 괄호: 사용자 1만 명당 인원, 평균 앞당긴 기간")

# 05 CT 판독 수준
ct = x["CT_판독수준"]
lv = ["현재", "PANDA", "CT 100%", "CT·조직검사\n100%"]
f, axs = plt.subplots(1, 2, figsize=(10, 5.6), dpi=200)
for ax, col, unit, ttl, c in ((axs[0], "증상 전 발견", "명", "증상 전 발견 (명)", BLUE), (axs[1], "생존연수", "년", "생존 기간 증가 (년)", BLUE)):
    v = ct[col].values
    bb = ax.bar(np.arange(4), v, 0.6, color=[c, c, c, c])
    for i, b in enumerate(bb):
        b.set_alpha(1.0 if i in (0, 3) else 0.75)
    bar_labels(ax, bb, lambda q: f"{q:.1f}", dy=max(v) * 0.015)
    ax.set_xticks(np.arange(4), lv, fontsize=10.5); ax.set_ylabel(ttl); ax.set_ylim(0, max(v) * 1.2); hgrid(ax)
f.subplots_adjust(wspace=0.25, left=0.07, right=0.98)
finish(f, "05_CT판독수준별_결과", "CT 판독이 좋아지면 같은 패치로 효과가 최대 1.6배가 된다",
       "사용자 1만 명당 · 판독 수준만 바꿔 같은 가상 환자에게 다시 적용 (PANDA: Cao 2023 Nat Med 외부 검증 민감도)", top=0.79)

# 06 CT 놓침 단계별
ms = x["CT_놓침_단계별"]
ok = ms["암 환자 CT 의뢰"] - ms["CT에서 놓침"] - ms["조직검사에서 놓침"]
slab = ["P0\n(CT에 안 보임)", "국한", "국소 진행", "원격 전이"]
f, ax = fig()
yi = np.arange(4)[::-1]
gap = 0.06
b1 = ax.barh(yi, ok, 0.6, color=BLUE, label="확진됨", edgecolor="white", linewidth=1.5)
b2 = ax.barh(yi, ms["CT에서 놓침"], 0.6, left=ok, color=ORANGE, label="CT에서 놓침", edgecolor="white", linewidth=1.5)
b3 = ax.barh(yi, ms["조직검사에서 놓침"], 0.6, left=ok + ms["CT에서 놓침"], color=AQUA, label="조직검사에서 놓침", edgecolor="white", linewidth=1.5)
for y, (_, r), o in zip(yi, ms.iterrows(), ok):
    tot = r["암 환자 CT 의뢰"]
    ax.text(tot + 0.15, y, f"의뢰 {tot:.1f}건 · CT 놓침 {r['CT에서 놓침']:.1f}건 ({100 * r['CT에서 놓침'] / tot:.0f}%)", va="center", fontsize=11, color=INK)
ax.set_yticks(yi, slab, fontsize=12); ax.set_xlabel("암 환자의 정밀검사 의뢰 (건, 사용자 1만 명당)"); ax.set_xlim(0, 17.5); vgrid(ax)
ax.legend(frameon=False, loc="lower right", fontsize=11)
tm = x["CT_판독수준"].iloc[0]
summ06 = (f"합계: 의뢰 {tm['암 환자 CT 의뢰(종양 있음)']:.1f}건 중 CT 놓침 {tm['CT에서 놓침']:.1f}건, 조직검사 놓침 {tm['조직검사에서 놓침']:.1f}건\n"
        f"양성이었지만 끝내 증상 전 확진 못 한 암 환자 {tm['양성이었지만 끝내 증상 전 확진 못 함(명)']:.1f}명")
f.subplots_adjust(left=0.16)
finish(f, "06_CT에서_놓친_횟수", "패치가 일찍 찾을수록 CT가 확인해 주지 못한다",
       "현재 판독 수준 · 정밀검사 시점의 종양 단계별 · 놓친 뒤에는 12개월 재검 제한에 걸림\n" + summ06, top=0.81)

# 07 시나리오: 생존 기간
sc = x["시나리오"].copy()
base = sc.loc[sc["시나리오"] == "기본"].iloc[0]
s2 = sc[sc["시나리오"] != "기본"].sort_values("생존연수")
f, ax = plt.subplots(figsize=(10, 6.4), dpi=200)
yi = np.arange(len(s2))
cols = [BLUE if v >= base["생존연수"] else ORANGE for v in s2["생존연수"]]
bb = ax.barh(yi, s2["생존연수"], 0.62, color=cols)
ax.axvline(base["생존연수"], color=INK, lw=1.2, ls="--")
for y, v in zip(yi, s2["생존연수"]):
    ax.text(v - 0.3, y, f"{v:.1f}", va="center", ha="right", fontsize=10.5, color="white", fontweight="bold")
ax.set_yticks(yi, s2["시나리오"], fontsize=10.5); ax.set_xlabel("늘어나는 생존 기간 합계 (년, 사용자 1만 명당)"); ax.set_xlim(0, 38); vgrid(ax)
from matplotlib.lines import Line2D
ax.legend(handles=[Line2D([0], [0], color=INK, lw=1.2, ls="--", label=f"기본 ({base['생존연수']:.1f}년)"), Patch(color=BLUE, label="기본보다 큼"), Patch(color=ORANGE, label="기본보다 작음")], frameon=False, loc="lower right", fontsize=10.5)
f.subplots_adjust(left=0.40, right=0.97)
finish(f, "07_시나리오별_생존기간", "가정을 하나씩 바꿨을 때: 늘어나는 생존 기간",
       "시나리오 표본(대조군 5만 명)이라 기본값이 본 분석(25.4년)과 약간 다름 · 숫자 표는 엑셀 07 시트", top=0.86)

# 08 시나리오: 증상 전 발견과 CT
s3 = sc.set_index("시나리오").loc[["기본"] + list(s2.sort_values("증상전발견")["시나리오"])]
f, axs = plt.subplots(1, 2, figsize=(10, 6.4), dpi=200, sharey=True)
yi = np.arange(len(s3))[::-1]
for ax, col, ttl, xm, fmt in ((axs[0], "증상전발견", "증상 전 발견 (명)", 24, "{:.1f}"), (axs[1], "CT", "추가 CT (건)", 3100, "{:,.0f}")):
    cc = [GRAY if n == "기본" else BLUE for n in s3.index]
    ax.barh(yi, s3[col], 0.62, color=cc)
    for y, v in zip(yi, s3[col]):
        ax.text(v + xm * 0.01, y, fmt.format(v), va="center", fontsize=9.5, color=INK)
    ax.set_xlabel(ttl); ax.set_xlim(0, xm); vgrid(ax)
axs[0].set_yticks(yi, s3.index, fontsize=9.5)
f.subplots_adjust(left=0.37, right=0.98, wspace=0.08)
finish(f, "08_시나리오별_발견과_CT", "가정을 하나씩 바꿨을 때: 증상 전 발견과 추가 CT",
       "사용자 1만 명당 · 회색 막대가 기본 · 재검 제한을 없애면 발견은 늘지만 CT가 2.2배", top=0.82)

# 09 같은 단계 이득 반영 비율
sv = sc.set_index("시나리오")
vals = [sv.loc["같은 단계 이득 0%", "생존연수"], sv.loc["기본", "생존연수"], sv.loc["같은 단계 이득 100%", "생존연수"]]
f, ax = fig()
bb = ax.bar([0, 1, 2], vals, 0.55, color=[GRAY, BLUE, GRAY])
bar_labels(ax, bb, lambda v: f"{v:.1f}년", dy=0.4, fs=12)
ax.set_xticks([0, 1, 2], ["0%", "50% (기본)", "100%"], fontsize=11.5)
ax.set_xlabel("같은 단계 이득 반영 비율"); ax.set_ylabel("늘어나는 생존 기간 (년, 사용자 1만 명당)"); ax.set_ylim(0, 36); hgrid(ax)
finish(f, "09_같은단계이득_반영비율", "같은 단계에서 일찍 찾았을 때의 이득을 얼마나 인정하나",
       "반영 비율은 직접 근거가 없는 가정이라 범위를 함께 보고 (시나리오 표본, 사용자 1만 명당)")

# 10 상관 민감도
cr = x["상관_민감도"].copy()
f, ax = fig()
yi = np.arange(len(cr))[::-1]
cc = [GRAY if i == 0 else BLUE for i in range(len(cr))]
ax.barh(yi, cr["생존연수"], 0.6, color=cc)
for y, v in zip(yi, cr["생존연수"]):
    ax.text(v + 0.3, y, f"{v:.1f}년", va="center", fontsize=11, color=INK)
ax.set_yticks(yi, [s.replace("-", "−") for s in cr["시나리오"]], fontsize=11.5); ax.set_xlim(0, 32)
ax.set_xlabel("늘어나는 생존 기간 (년, 사용자 1만 명당)"); vgrid(ax)
f.subplots_adjust(left=0.30)
finish(f, "10_혈당지방_상관_민감도", "혈당-피하지방 상관을 크게 바꿔도 결과는 거의 같다 (25.3~26.1년)",
       "상관 = 암 환자끼리 비교했을 때 혈당이 많이 오른 사람이 지방도 많이 빠지는 정도 · 회색이 기본")

# 11 당뇨 초과 사망
dm = [("배수 없음\n(일반인 생명표)", sv.loc["당뇨 초과 사망 없음 (일반인 생명표)", "생존연수"]),
      ("1.92 / 1.40 / 1.11 / 0.95\n(기본)", sv.loc["기본", "생존연수"]),
      ("2.00 / 1.55 / 1.25 / 1.03", sv.loc["당뇨 초과 사망 큼 (당화혈색소 7.0~7.8% 행)", "생존연수"])]
f, ax = fig()
bb = ax.bar([0, 1, 2], [v for _, v in dm], 0.55, color=[GRAY, BLUE, GRAY])
bar_labels(ax, bb, lambda v: f"{v:.1f}년", dy=0.4, fs=12)
ax.set_xticks([0, 1, 2], [n for n, _ in dm], fontsize=11.5); ax.set_xlabel("당뇨 환자 사망 배수 (55세 미만 / 55~64 / 65~74 / 75세 이상)"); ax.set_ylabel("늘어나는 생존 기간 (년, 사용자 1만 명당)"); ax.set_ylim(0, 32); hgrid(ax)
finish(f, "11_당뇨_배경사망_배수", "당뇨 환자의 초과 사망을 반영해도 결과는 거의 같다",
       "다른 원인 사망률을 당뇨 환자 기준으로 바꾼 비교 (시나리오 표본, 사용자 1만 명당)")

# 12 착용 일정 × 재검 제한
sd = pd.read_csv(SCHED)
nm = [f"{a}\n재검 제한 {b}" for a, b in zip(sd["착용 일정"], sd["재검 제한"])]
f, axs = plt.subplots(1, 2, figsize=(10, 5.6), dpi=200)
for ax, col, ttl, fmt in ((axs[0], "생존 기간 증가(년)", "늘어나는 생존 기간 (년)", "{:.1f}"), (axs[1], "CT", "추가 CT (건)", "{:,.0f}")):
    cc = [BLUE if "기본" in a else GRAY for a in sd["착용 일정"]]
    yi = np.arange(len(sd))[::-1]
    ax.barh(yi, sd[col], 0.6, color=cc)
    for y, v in zip(yi, sd[col]):
        ax.text(v + sd[col].max() * 0.015, y, fmt.format(v), va="center", fontsize=10.5, color=INK)
    ax.set_xlabel(ttl); ax.set_xlim(0, sd[col].max() * 1.22); vgrid(ax)
    ax.set_yticks(yi, nm if ax is axs[0] else [""] * len(sd), fontsize=10)
f.subplots_adjust(left=0.24, right=0.98, wspace=0.08)
finish(f, "12_착용일정_재검제한", "착용 일정과 재검 제한 규칙에 따른 효과와 CT 부담",
       "현재 모델로 다시 계산 (사용자 1만 명당, 10만 명 코호트) · 파란 막대가 기본 · 비용은 엑셀 12 시트", top=0.80)

# 13 전국 연간
na = x["전국_연간"].set_index("항목")
items13 = [("증상 전 발견 (명)", "증상 전 발견", 1), ("국한(초기) 진단 증가 (명)", "국한(초기) 진단", 1), ("4기 진단 감소 (명)", "원격 전이(4기) 진단", -1),
           ("생존 기간 증가 (년)", "생존연수", 1), ("QALY 증가", "QALY", 1), ("추가 CT (건)", "CT", 1),
           ("암이 확인되지 않은 정밀검사 (회)", "암 미확인 정밀검사", 1), ("사회적 가치 (억 원)", na.index[-1], 1)]
rows13 = [[a, f"{sg * na.loc[k, '기본']:,.0f}", f"{sg * na.loc[k, '하한']:,.0f} ~ {sg * na.loc[k, '상한']:,.0f}"] for a, k, sg in items13]
table_img("13_전국_연간효과", ["항목 (1년)", "15만 명", "10만 ~ 25만 명"], rows13, [0.5, 0.2, 0.3])

# 14 1인당 비용과 사회적 가치
wear = e.loc["착용 횟수", "패치를 쓸 때"] / 1e4
dev = wear * 7 + 2; tot = e.loc["총비용(만원)", "차이"] / 1e4; oth = tot - dev
val = e.loc["QALY", "차이"] / 1e4 * 5000; vald = e.loc["QALY (할인 4.5%)", "차이"] / 1e4 * 5000
f, ax = fig()
ax.barh([1], [dev], 0.5, color=BLUE, label="기기", edgecolor="white", linewidth=1.5)
ax.barh([1], [oth], 0.5, left=[dev], color=ORANGE, label="추가 검사·진료비", edgecolor="white", linewidth=1.5)
ax.barh([0], [val], 0.5, color=AQUA, label="사회적 가치")
ax.text(tot + 1, 1, f"{tot:.1f}만 원  (기기 {dev:.1f} + 검사·진료 {oth:.1f})", va="center", fontsize=11.5, color=INK)
ax.text(val + 1, 0, f"{val:.1f}만 원  (할인 4.5% 적용 시 {vald:.1f}만 원)", va="center", fontsize=11.5, color=INK)
ax.set_yticks([1, 0], ["1인당 추가 비용", "1인당 사회적 가치"], fontsize=12.5); ax.set_xlim(0, 125); ax.set_xlabel("만 원 (사용자 1인당, 3년)"); vgrid(ax)
ax.legend(frameon=False, loc="lower right", fontsize=10.5)
f.subplots_adjust(left=0.2)
finish(f, "14_1인당_비용과_가치", "현재 가격 가정에서는 비용(대부분 기기)이 사회적 가치보다 크다",
       "착용 1회 7만 원, 허브 2만 원은 팀이 기기 구성을 정한 뒤 수정할 가정값")

# 15 PSA
raw = x["PSA_원자료"]; ps = x["PSA_요약"].set_index(x["PSA_요약"].columns[0])
f, ax = fig()
ax.hist(raw["생존연수"], bins=np.arange(8, 47, 2.5), color=GRAY, edgecolor="white", linewidth=1.5)
med, lo, hi = ps.loc["생존연수", "중앙값"], ps.loc["생존연수", "2.5%"], ps.loc["생존연수", "97.5%"]
ax.set_ylim(0, 14); ax.axvline(med, color=BLUE, lw=2); ax.axvline(lo, color=BLUE, lw=1.2, ls=":"); ax.axvline(hi, color=BLUE, lw=1.2, ls=":")
ax.text(med + 0.4, ax.get_ylim()[1] * 0.95, f"중앙값 {med:.1f}년", color=INK, fontsize=11.5, va="top")
ax.text(lo + 0.4, ax.get_ylim()[1] * 0.80, f"2.5% {lo:.1f}년", color=INK2, fontsize=10.5, va="top")
ax.text(hi - 0.4, ax.get_ylim()[1] * 0.80, f"97.5% {hi:.1f}년", color=INK2, fontsize=10.5, va="top", ha="right")
ax.set_xlabel("늘어나는 생존 기간 합계 (년, 사용자 1만 명당)"); ax.set_ylabel("반복 횟수 (60회 중)"); hgrid(ax)
finish(f, "15_확률민감도분석", "불확실한 입력값을 모두 바꿔 60번 반복해도 효과는 양수다",
       "입력값 23개와 자연경과 조합(107개 중 하나)을 무작위로 뽑음 · 중앙값이 기본(25.4년)보다 낮은 것은 이득 실현 비율을 50~100%에서 뽑았기 때문")

# 16 완벽한 기기 대비
hr = x["완벽한기기_대비"]
perfect, ours, ratio = hr.iloc[0, 1], hr.iloc[1, 1], hr.iloc[2, 1]
f, ax = fig()
bb = ax.barh([1, 0], [perfect, ours], 0.5, color=[GRAY, BLUE])
ax.text(perfect + 1, 1, f"{perfect:.1f}", va="center", fontsize=12, color=INK)
ax.text(ours + 1, 0, f"{ours:.1f}  ({ratio:.0f}%)", va="center", fontsize=12, color=INK, fontweight="bold")
ax.set_yticks([1, 0], ["완벽한 기기", "우리 패치"], fontsize=12); ax.set_xlim(0, 85)
ax.set_xlabel("QALY 증가 (사용자 1만 명당, 할인 4.5%)"); vgrid(ax)
f.subplots_adjust(left=0.18)
finish(f, "16_완벽한기기_대비", f"우리 패치는 이론적 최대 효과의 약 {ratio:.0f}%를 실현한다",
       "같은 자연경과에서 가능한 최대치와 비교 · 차이의 상당 부분은 CT 확진 단계의 한계 (그림 05)")

# 17 검증
vd = x["검증"]
f, axs = plt.subplots(1, 2, figsize=(10, 5.6), dpi=200)
t = vd[vd["구분"] == "진단 시점"]; s = vd[vd["구분"] == "진단 병기(S0)"]
for ax, d, ttl, lb in ((axs[0], t, "신규 당뇨 후 진단 시점 (%)", ["0~4", "4~12", "12~24", "24~36개월"]),
                       (axs[1], s, "진단 병기 분포 (%)", ["국한", "국소", "원격"])):
    xi = np.arange(len(d))
    b0 = ax.bar(xi - 0.19, d["목표"] * 100, 0.36, color=GRAY, label="실제 자료")
    b1 = ax.bar(xi + 0.19, d["모델"] * 100, 0.36, color=BLUE, label="모델")
    bar_labels(ax, b0, lambda v: f"{v:.0f}", dy=0.6, fs=9.5); bar_labels(ax, b1, lambda v: f"{v:.0f}", dy=0.6, fs=9.5)
    ax.set_xticks(xi, lb, fontsize=10.5); ax.set_ylabel(ttl); hgrid(ax)
    ax.set_ylim(0, max(d["목표"].max(), d["모델"].max()) * 100 * 1.25)
axs[0].legend(frameon=False, fontsize=10.5, loc="upper right")
f.subplots_adjust(wspace=0.2, left=0.06, right=0.98)
finish(f, "17_모델검증", "모델은 실제 진단 시점과 병기 분포를 재현한다",
       "패치 없이 돌린 결과와 실제 통계 비교 · 진단 시점: Chari 2025 · 병기: 국가암등록통계", top=0.79)

print("\n".join(made))
