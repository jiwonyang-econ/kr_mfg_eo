"""
07b. 관세 이벤트 스터디 v2 (07의 1차 결과를 보완)

v1 대비 바뀐 점
  1. 본사 주소 수출 제거: 공장 등록 자료에 자동차(KSIC 30) 종업원이 MIN_AUTO_EMP 미만인 시군구 제외
  2. 정책 구간별 합동 추정 (H1 검정용 핵심 표)
       U    2024-11 ~ 2025-03  대선·관세 위협 ~ 시행 전 (불확실성)
       T25a 2025-04 ~ 2025-07  25% 시행 ~ 한미 합의 전
       T25b 2025-08 ~ 2025-10  합의 발표 후, 세율은 아직 25%
       T15  2025-11 ~ 2026-01  15% 적용
       P26  2026-02 ~          대법원 판결 이후
     기준 = 2022-01 ~ 2024-10
  3. 사전 추세 통제: 노출도 × 선형 추세를 넣은 사양 추가
  4. 외연(수출 여부 0/1)과 내연(로그 금액) 분리
  5. 추론: HS6(설계 A)·시군구(설계 B) 군집 + wild cluster bootstrap p값 (Rademacher, 귀무가설 부과, 999회)
  6. 이벤트 스터디 그림을 기준월 2024-10(대선 전)으로도 그림

출력: data/output/tariff_v2/ (pooled.csv, es_*.png, report_v2.md, kept_regions.csv)
"""
import sys
import numpy as np
import pandas as pd
from config import RAW, INT, OUT

DST = OUT / "tariff_v2"
DST.mkdir(parents=True, exist_ok=True)
VEH4 = {"8703", "8704"}
MIN_AUTO_EMP = 100
MAX_EXP_PER_EMP = 5e6       # 2024년 자동차 종업원 1인당 수출이 이보다 크면 본사·무역법인 주소로 보고 제외
START = pd.Timestamp("2022-01-01")
PERIODS = [("U", "2024-11-01", "2025-03-01"), ("T25a", "2025-04-01", "2025-07-01"),
           ("T25b", "2025-08-01", "2025-10-01"), ("T15", "2025-11-01", "2026-01-01"),
           ("P26", "2026-02-01", "2099-12-01")]
PLAB = {"U": "불확실성 (24.11~25.3)", "T25a": "25% 시행 (25.4~7)", "T25b": "합의 후·25% (25.8~10)",
        "T15": "15% 적용 (25.11~26.1)", "P26": "판결 이후 (26.2~)"}
EVENTS = {"2024-11-01": "US election (Nov 5)", "2025-03-01": "Sec.232 proclaimed (Mar 26)",
          "2025-04-01": "Autos 25%", "2025-05-01": "Parts 25%", "2025-07-01": "Deal (Jul 30)",
          "2025-11-01": "15% applied", "2026-02-01": "SCOTUS (Feb 20)"}
GWANGJU_GU = {"광산구", "북구", "서구", "남구", "동구"}
SHORT = [("서울", "서울"), ("부산", "부산"), ("대구", "대구"), ("인천", "인천"), ("광주", "광주"),
         ("대전", "대전"), ("울산", "울산"), ("세종", "세종"), ("경기", "경기"), ("강원", "강원"),
         ("충청북", "충북"), ("충북", "충북"), ("충청남", "충남"), ("충남", "충남"), ("전라북", "전북"),
         ("전북", "전북"), ("전라남", "전남"), ("전남", "전남"), ("경상북", "경북"), ("경북", "경북"),
         ("경상남", "경남"), ("경남", "경남"), ("제주", "제주")]
rng = np.random.default_rng(20251101)


# ---------------------------------------------------------------- 자료
def addr_region(a):
    t = str(a).split()
    if len(t) < 2:
        return None
    if t[0].startswith("전남광주"):
        return ("광주" if t[1] in GWANGJU_GU else "전남") + " " + t[1]
    for k, v in SHORT:
        if t[0].startswith(k):
            return v + " " + t[1]
    return None


def auto_regions():
    f = INT / "factories_geocoded.csv"
    fac = pd.read_csv(f, low_memory=False, usecols=lambda c: c in ("geo_addr", "addr_road", "ksic2", "employees"))
    fac = fac[pd.to_numeric(fac["ksic2"], errors="coerce") == 30]
    addr = fac["geo_addr"].where(fac["geo_addr"].notna(), fac.get("addr_road"))
    fac["r"] = addr.map(addr_region)
    emp = pd.to_numeric(fac["employees"], errors="coerce").fillna(0)
    return emp.groupby(fac["r"]).sum()


def load():
    us = pd.read_csv(RAW / "customs" / "nitem_trade.csv", dtype={"hs": str}, parse_dates=["ym"])
    us = us[us["hs"].str.len() == 6]
    sg = pd.read_csv(RAW / "customs_sgg" / "sgg_hs_trade.csv", dtype={"hsSgn": str}, parse_dates=["ym"])
    sg = sg.rename(columns={"hsSgn": "hs"})
    sg["exp"] = pd.to_numeric(sg["expUsdAmt"], errors="coerce").fillna(0) * 1000
    sg["r"] = sg["sido"].astype(str) + " " + sg["sgg"].astype(str).str.split().str[0]
    sg = sg.groupby(["r", "sido", "hs", "ym"], as_index=False)["exp"].sum()
    return us, sg


def us_share(us, sg):
    nat = sg[sg["ym"].dt.year == 2024].groupby("hs")["exp"].sum().rename("exp_kr")
    u = us[us["ym"].dt.year == 2024].groupby("hs")["expDlr"].sum().rename("exp_us")
    s = pd.concat([u, nat], axis=1).dropna(subset=["exp_kr"]).fillna({"exp_us": 0})
    s["s_us"] = (s["exp_us"] / s["exp_kr"]).clip(0, 1)
    s["group"] = np.where(s.index.str[:4].isin(VEH4), "VEH", "PARTS")
    return s


# ---------------------------------------------------------------- 추정 도구
def demean(df, cols, fes, iters=300, tol=1e-9):
    x = df[cols].astype(float).copy()
    for _ in range(iters):
        prev = x.values.copy()
        for f in fes:
            x = x - x.groupby(df[f]).transform("mean")
        if np.nanmax(np.abs(x.values - prev)) < tol:
            break
    return x


class Cl:
    """군집 강건 OLS를 빠르게 반복하기 위한 준비 (X·군집 고정, y만 바뀜)."""
    def __init__(self, X, g):
        self.order = np.argsort(g, kind="stable")
        gs = g[self.order]
        self.starts = np.r_[0, np.flatnonzero(np.diff(gs)) + 1]
        self.X = X
        self.Xs = X[self.order]
        self.XtX_inv = np.linalg.pinv(X.T @ X)
        n, k = X.shape
        G = self.starts.size
        self.c = G / (G - 1) * (n - 1) / (n - k)
        self.G = G

    def fit(self, y):
        b = self.XtX_inv @ (self.X.T @ y)
        u = y - self.X @ b
        S = np.add.reduceat(self.Xs * u[self.order][:, None], self.starts, axis=0)
        V = self.c * self.XtX_inv @ (S.T @ S) @ self.XtX_inv
        return b, np.sqrt(np.diag(V)), u


def ols_cl(y, X, g):
    return Cl(X, g).fit(y)


def wild_p(y, X, g, j, reps=999):
    """귀무가설(β_j = 0)을 부과한 wild cluster bootstrap (Rademacher) p값."""
    full = Cl(X, g)
    b, se, _ = full.fit(y)
    t0 = b[j] / se[j]
    Xr = np.delete(X, j, axis=1)
    br = np.linalg.pinv(Xr.T @ Xr) @ (Xr.T @ y)
    fit, ur = Xr @ br, y - Xr @ br
    _, inv = np.unique(g, return_inverse=True)
    cnt = 0
    for _ in range(reps):
        w = rng.choice([-1.0, 1.0], size=full.G)[inv]
        bs, ses, _ = full.fit(fit + w * ur)
        cnt += abs(bs[j] / ses[j]) >= abs(t0)
    return (cnt + 1) / (reps + 1)


def pooled(df, y, inten, fes, cluster, trend=False, boot=True):
    d = df.dropna(subset=[y]).copy()
    cols = []
    for p, a, b in PERIODS:
        c = f"P_{p}"
        d[c] = ((d["ym"] >= a) & (d["ym"] <= b)).astype(float) * d[inten]
        if d[c].abs().sum() > 0:
            cols.append(c)
    if trend:
        d["trend"] = d[inten] * ((d["ym"] - START).dt.days / 365.25)
        cols.append("trend")
    z = demean(d, [y] + cols, fes)
    Y, X = z[y].values, z[cols].values
    g = d[cluster].astype("category").cat.codes.values
    b, se, _ = ols_cl(Y, X, g)
    rows = []
    for j, c in enumerate(cols):
        if c == "trend":
            continue
        p = wild_p(Y, X, g, j) if boot else np.nan
        rows.append(dict(period=c[2:], b=b[j], se=se[j], p_wild=p, n=len(d), clusters=np.unique(g).size))
    return pd.DataFrame(rows)


def kof(ym, base):
    return (ym.dt.year - base.year) * 12 + ym.dt.month - base.month - 1


def event_study(df, y, inten, fes, cluster, base, kmin=-18, kmax=None):
    d = df.dropna(subset=[y]).copy()
    k = kof(d["ym"], base)
    kmax = min(kmax or 99, int(k.max()))
    k = k.clip(kmin, kmax)
    cols = []
    for kk in range(kmin, kmax + 1):
        if kk == -1:
            continue
        c = f"E_{kk}"
        d[c] = (k == kk).astype(float) * d[inten]
        cols.append(c)
    z = demean(d, [y] + cols, fes)
    b, se, _ = ols_cl(z[y].values, z[cols].values, d[cluster].astype("category").cat.codes.values)
    t = pd.DataFrame({"k": [int(c[2:]) for c in cols], "b": b, "se": se})
    t = pd.concat([t, pd.DataFrame({"k": [-1], "b": [0.0], "se": [0.0]})]).sort_values("k")
    t["lo"], t["hi"] = t["b"] - 1.96 * t["se"], t["b"] + 1.96 * t["se"]
    return t


def plot_es(tabs, base, title, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(9.5, 4.6))
    for (lab, t), c in zip(tabs.items(), ["#1f5aa6", "#c4572a", "#2f8f5b"]):
        ax.plot(t["k"], t["b"], marker="o", ms=3, color=c, label=lab)
        ax.fill_between(t["k"], t["lo"], t["hi"], color=c, alpha=.13)
    ax.axhline(0, color="k", lw=.8)
    lo, hi = ax.get_xlim()
    for d, lab in EVENTS.items():
        kk = kof(pd.Series([pd.Timestamp(d)]), base).iloc[0]
        if lo < kk < hi:
            ax.axvline(kk, color="gray", ls=":", lw=.7)
            ax.text(kk, ax.get_ylim()[1], lab, rotation=90, va="top", ha="right", fontsize=7, color="gray")
    ax.set_xlabel(f"months relative to {base:%Y-%m} (k=-1)")
    ax.set_title(title, fontsize=10)
    ax.legend(fontsize=8, loc="lower left")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------- 본 분석
def main(boot=True):
    us, sg = load()
    s = us_share(us, sg)
    ae = auto_regions()
    allr = sg.groupby("r")["exp"].sum().sort_values(ascending=False)
    x24 = sg[sg["ym"].dt.year == 2024].groupby("r")["exp"].sum()
    per = x24 / ae.reindex(x24.index).fillna(0).clip(lower=1)
    hq = set(per[per > MAX_EXP_PER_EMP].index)
    keep_r = set(ae[ae >= MIN_AUTO_EMP].index) - hq
    print("본사형으로 제외(2024 수출/자동차 종업원, 백만 달러):",
          per[per.index.isin(hq)].sort_values(ascending=False).head(10).div(1e6).round(1).to_dict())
    kept = pd.DataFrame({"exp_total": allr, "auto_emp": allr.index.map(ae).fillna(0),
                         "exp2024_per_emp": allr.index.map(per)})
    kept["kept"] = kept.index.isin(keep_r)
    kept.to_csv(DST / "kept_regions.csv", encoding="utf-8-sig")
    drop_share = kept.loc[~kept["kept"], "exp_total"].sum() / kept["exp_total"].sum()
    print(f"자동차 종업원 {MIN_AUTO_EMP}명 이상 시군구 {len(keep_r)}개 유지, 제외된 수출 비중 {drop_share:.1%}")
    print("제외된 상위 시군구:", kept[~kept["kept"]].head(8)["exp_total"].div(1e9).round(1).to_dict())
    sg = sg[sg["r"].isin(keep_r) & (sg["ym"] >= START)]

    # 설계 A 패널 (지역·품목 균형, 0 채움)
    base = sg[sg["ym"].dt.year == 2024].groupby(["r", "hs"])["exp"].sum()
    pairs = base[base > 1e6].index
    months = sorted(sg["ym"].unique())
    a = sg.set_index(["r", "hs", "ym"])["exp"]
    a = a.reindex(pd.MultiIndex.from_tuples([(r, h, t) for r, h in pairs for t in months],
                                            names=["r", "hs", "ym"]), fill_value=0).reset_index()
    a = a.merge(s[["s_us", "group"]], left_on="hs", right_index=True).dropna(subset=["s_us"])
    a["sido"] = a["r"].str.split().str[0]
    a["y_log"] = np.log(a["exp"].where(a["exp"] > 0))
    a["y_any"] = (a["exp"] > 0).astype(float)
    a["rh"], a["rt"] = a["r"] + "|" + a["hs"], a["r"] + "|" + a["ym"].astype(str)

    # 설계 B 패널
    w = sg[sg["ym"].dt.year == 2024].groupby(["r", "hs"])["exp"].sum().reset_index()
    w = w.merge(s[["s_us"]], left_on="hs", right_index=True)
    w["wx"] = w["exp"] * w["s_us"]
    E = w.groupby("r").agg(exp2024=("exp", "sum"), us2024=("wx", "sum"))
    E["E"] = E["us2024"] / E["exp2024"]
    E = E[E["exp2024"] > 5e6]
    b = sg.groupby(["r", "ym"])["exp"].sum().reindex(
        pd.MultiIndex.from_product([E.index, months], names=["r", "ym"]), fill_value=0).reset_index()
    b["sido"] = b["r"].str.split().str[0]
    b["E"] = b["r"].map(E["E"])
    b["y_log"] = np.log(b["exp"].where(b["exp"] > 0))
    b["st"] = b["sido"] + "|" + b["ym"].astype(str)

    # ---- 합동 추정
    specs = [("A 전체·로그", a, "y_log", "s_us", ["rh", "rt"], "hs", False),
             ("A 완성차·로그", a[a["group"] == "VEH"], "y_log", "s_us", ["rh", "rt"], "hs", False),
             ("A 부품·로그", a[a["group"] == "PARTS"], "y_log", "s_us", ["rh", "rt"], "hs", False),
             ("A 부품·로그·추세", a[a["group"] == "PARTS"], "y_log", "s_us", ["rh", "rt"], "hs", True),
             ("A 전체·로그·추세", a, "y_log", "s_us", ["rh", "rt"], "hs", True),
             ("A 전체·수출여부(0/1)", a, "y_any", "s_us", ["rh", "rt"], "hs", False),
             ("B 시군구·로그", b, "y_log", "E", ["r", "st"], "r", False),
             ("B 시군구·로그·추세", b, "y_log", "E", ["r", "st"], "r", True)]
    res = []
    for lab, d, y, inten, fes, cl, tr in specs:
        if d[cl].nunique() < 5:
            continue
        r = pooled(d, y, inten, fes, cl, trend=tr, boot=boot)
        r.insert(0, "spec", lab)
        res.append(r)
        print(f"  완료: {lab}")
    res = pd.concat(res)
    res.to_csv(DST / "pooled.csv", index=False, encoding="utf-8-sig")

    # ---- 이벤트 스터디 그림 (기준월 2025-03, 2024-10)
    for bm in ("2025-03-01", "2024-10-01"):
        bt = pd.Timestamp(bm)
        tabs = {g: event_study(a if g == "ALL" else a[a["group"] == g], "y_log", "s_us", ["rh", "rt"], "hs", bt)
                for g in ("ALL", "VEH", "PARTS")}
        plot_es(tabs, bt, f"A: product-level, US share (Seoul/HQ regions removed), base {bt:%Y-%m}",
                DST / f"es_product_base{bt:%Y%m}.png")
        tb = event_study(b, "y_log", "E", ["r", "st"], "r", bt)
        plot_es({"region exposure": tb}, bt, f"B: municipality exposure, base {bt:%Y-%m}",
                DST / f"es_region_base{bt:%Y%m}.png")

    # ---- 보고서
    def star(p):
        return "***" if p < .01 else "**" if p < .05 else "*" if p < .1 else ""
    pcols = [p for p, _, _ in PERIODS]
    lines = ["# 관세 이벤트 스터디 v2", "",
             f"- 유지 시군구 {len(keep_r)}개 (자동차 종업원 {MIN_AUTO_EMP}명 이상), 제외 수출 비중 {drop_share:.1%}",
             f"- 설계 A: 지역·품목 {a['rh'].nunique():,}개, 품목 {a['hs'].nunique()}개 / 설계 B: 시군구 {len(E)}개",
             f"- 기준 기간 {START:%Y-%m} ~ 2024-10. 괄호는 군집 표준오차, 별표는 wild cluster bootstrap p값 기준", "",
             "| 사양 | " + " | ".join(PLAB[p] for p in pcols) + " |", "|---|" + "---|" * len(pcols)]
    for spec, g in res.groupby("spec", sort=False):
        cells = []
        for p in pcols:
            x = g[g["period"] == p]
            cells.append("" if x.empty else f"{x.b.iloc[0]:.3f}{star(x.p_wild.iloc[0])} ({x.se.iloc[0]:.3f})")
        lines.append(f"| {spec} | " + " | ".join(cells) + " |")
    lines += ["", "읽는 법 (H1): '불확실성' 구간 계수가 음수면 시행 전 반응(발표·불확실성 효과).",
              "'합의 후·25%'가 '25% 시행'보다 덜 음수면 세율이 그대로인데 합의만으로 회복 → 불확실성 해소 효과.",
              "'15% 적용'에서 추가 회복이 있으면 관세 수준 효과."]
    (DST / "report_v2.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main(boot="--noboot" not in sys.argv)
