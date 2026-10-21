"""
07. 2025년 미국 자동차 관세(232조) 노출도와 이벤트 스터디 (1차 버전)

입력
  data/raw/customs/nitem_trade.csv          00d: 대미 수출, HS6 × 월 (달러)
  data/raw/customs_sgg/sgg_hs_trade.csv     00e: 시군구 × HS6 × 월 수출 (천 달러, 상대국 구분 없음)
출력 (data/output/tariff/)
  hs6_us_share.csv        HS6별 대미 비중(2024) = 대미 수출 / 전국 수출(시군구 합)
  sgg_exposure.csv        시군구별 노출도 E_r = Σ_h (지역 내 품목 비중, 2024) × 대미 비중_h
  es_product.csv/.png     설계 A: 같은 지역·같은 달 안에서 대미 비중이 큰 품목이 더 줄었나
  es_region.csv/.png      설계 B: 노출도가 큰 시군구의 자동차 공급망 수출이 더 줄었나
  us_exports_monthly.png  대미 완성차·부품 수출 월별 추이 + 이벤트 표시
  tariff_report.md

추정식
  A: ln(X_rht) = Σ_k β_k · 1[t = 기준월+k] · s_h + α_rh + γ_rt + ε    (군집: HS6)
     → 지역 공통 충격(γ_rt)을 빼고, 품목 간 대미 노출 차이로 식별
  B: ln(Y_rt)     = Σ_k β_k · 1[t = 기준월+k] · E_r + α_r + γ_(시도)t + ε (군집: 시군구)
  기준월 = 2025-03 (k = -1), 사건 창 = -24 ~ +17개월 (양 끝은 묶음)

주의: 232조 부품 대상 목록(Annex)은 HS8 단위라 HS6 대응은 근사. 대상 여부를 엄밀히 하려면
      공식 부속서로 VEH/PARTS 분류를 다시 확인할 것.
"""
import numpy as np
import pandas as pd
import statsmodels.api as sm
from config import RAW, OUT

DST = OUT / "tariff"
DST.mkdir(parents=True, exist_ok=True)
BASE = pd.Timestamp("2025-03-01")        # k = -1 (발표 3/26, 완성차 발효 4/3 직전 달)
KMIN, KMAX = -24, 17
EVENTS = {                               # 그림 표시용
    "2025-04-01": "Autos 25% (Apr 3)",
    "2025-05-01": "Parts 25% (May 3)",
    "2025-07-01": "KR-US deal (Jul 30)",
    "2025-11-01": "Autos 15% (Nov 1)",
    "2026-02-01": "SCOTUS IEEPA (Feb 20)",
}
VEH4 = {"8703", "8704"}                  # 완성차 (4/3 발효); 나머지 공급망 HS = 부품 (5/3)


def load():
    us = pd.read_csv(RAW / "customs" / "nitem_trade.csv", dtype={"hs": str}, parse_dates=["ym"])
    us = us[us["hs"].str.len() == 6]
    sg = pd.read_csv(RAW / "customs_sgg" / "sgg_hs_trade.csv", dtype={"hsSgn": str}, parse_dates=["ym"])
    sg = sg.rename(columns={"hsSgn": "hs"})
    sg["exp"] = pd.to_numeric(sg["expUsdAmt"], errors="coerce").fillna(0) * 1000      # 천 달러 → 달러
    sg["r"] = sg["sido"].astype(str) + " " + sg["sgg"].astype(str)
    return us, sg


def us_share(us, sg, year=2024):
    nat = sg[sg["ym"].dt.year == year].groupby("hs")["exp"].sum().rename("exp_kr")
    u = us[us["ym"].dt.year == year].groupby("hs").agg(exp_us=("expDlr", "sum"), name=("item_name", "first"))
    s = u.join(nat, how="right").fillna({"exp_us": 0})
    s["s_us"] = (s["exp_us"] / s["exp_kr"]).clip(0, 1)
    s["group"] = np.where(s.index.str[:4].isin(VEH4), "VEH", "PARTS")
    s = s.sort_values("exp_kr", ascending=False)
    s.to_csv(DST / "hs6_us_share.csv", encoding="utf-8-sig")
    return s


def demean(df, cols, fes, iters=200, tol=1e-9):
    x = df[cols].astype(float).copy()
    for _ in range(iters):
        prev = x.values.copy()
        for f in fes:
            x = x - x.groupby(df[f]).transform("mean")
        if np.nanmax(np.abs(x.values - prev)) < tol:
            break
    return x


def kof(ym):
    """사건 시점 k: BASE(2025-03) = -1, 2025-04 = 0."""
    return (ym.dt.year - BASE.year) * 12 + ym.dt.month - BASE.month - 1


def event_dummies(df, intensity, prefix):
    global KMAX
    KMAX = min(KMAX, int(kof(df["ym"]).max()))       # 자료 마지막 달까지만
    k = (kof(df["ym"])).clip(KMIN, KMAX)
    cols = []
    for kk in range(KMIN, KMAX + 1):
        if kk == -1:
            continue
        c = f"{prefix}{kk}"
        df[c] = (k == kk).astype(float) * df[intensity]
        cols.append(c)
    return cols


def run_es(df, y, cols, fes, cluster):
    z = demean(df, [y] + cols, fes)
    m = sm.OLS(z[y], z[cols]).fit(cov_type="cluster", cov_kwds={"groups": df[cluster].astype("category").cat.codes})
    out = pd.DataFrame({"k": [int(c.split("_")[-1]) for c in cols], "b": m.params.values, "se": m.bse.values})
    out = pd.concat([out, pd.DataFrame({"k": [-1], "b": [0.0], "se": [0.0]})]).sort_values("k")
    out["lo"], out["hi"] = out["b"] - 1.96 * out["se"], out["b"] + 1.96 * out["se"]
    return out, int(m.nobs)


def plot_es(tabs, title, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(9, 4.5))
    for (lab, t), c in zip(tabs.items(), ["#1f5aa6", "#c4572a", "#2f8f5b"]):
        ax.plot(t["k"], t["b"], marker="o", ms=3, color=c, label=lab)
        ax.fill_between(t["k"], t["lo"], t["hi"], color=c, alpha=.15)
    ax.axhline(0, color="k", lw=.8)
    ax.axvline(-0.5, color="gray", ls="--", lw=.8)
    for d, lab in EVENTS.items():
        kk = kof(pd.Series([pd.Timestamp(d)])).iloc[0]
        if KMIN < kk < KMAX:
            ax.axvline(kk, color="gray", ls=":", lw=.7)
            ax.text(kk, ax.get_ylim()[1], lab, rotation=90, va="top", ha="right", fontsize=7, color="gray")
    ax.set_xlabel("months relative to 2025-03 (k=-1)")
    ax.set_title(title, fontsize=10)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_us(us, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    u = us.assign(g=np.where(us["hs"].str[:4].isin(VEH4), "Vehicles (8703-04)", "Parts & components"))
    t = u.groupby(["ym", "g"])["expDlr"].sum().unstack() / 1e9
    fig, ax = plt.subplots(figsize=(9, 4))
    t.plot(ax=ax, color=["#c4572a", "#1f5aa6"][: t.shape[1]])
    for d, lab in EVENTS.items():
        ax.axvline(pd.Timestamp(d), color="gray", ls=":", lw=.7)
        ax.text(pd.Timestamp(d), ax.get_ylim()[1], lab, rotation=90, va="top", ha="right", fontsize=7, color="gray")
    ax.set_ylabel("US-bound exports, USD bn / month")
    ax.set_xlabel("")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    us, sg = load()
    s = us_share(us, sg)
    print(f"HS6 {len(s)}개, 대미 비중(2024, 수출액 가중) {np.average(s['s_us'].fillna(0), weights=s['exp_kr'].fillna(0)):.1%}")
    print(s.head(12)[["name", "group", "exp_kr", "exp_us", "s_us"]].to_string())

    # ---- 설계 A: 지역×품목×월 ----
    a = sg.groupby(["r", "sido", "hs", "ym"], as_index=False)["exp"].sum()
    base = a[a["ym"].dt.year == 2024].groupby(["r", "hs"])["exp"].sum()
    keep = base[base > 1e6].index                          # 2024년 100만 달러 이상 수출한 지역·품목만
    a = a.set_index(["r", "hs"]).loc[lambda d: d.index.isin(keep)].reset_index()
    # 빠진 달은 0 수출로 채움 (관세청 자료는 수출 없으면 행이 없음)
    full = pd.MultiIndex.from_product([a[["r", "hs"]].drop_duplicates().apply(tuple, axis=1).tolist(),
                                       sorted(a["ym"].unique())])
    a = a.set_index(["r", "hs", "ym"])["exp"]
    a = a.reindex([(r, h, t) for (r, h), t in full], fill_value=0).reset_index()
    a["sido"] = a["r"].str.split().str[0]
    a = a.merge(s[["s_us", "group"]], left_on="hs", right_index=True, how="left").dropna(subset=["s_us"])
    a["y"] = np.log(a["exp"].where(a["exp"] > 0))          # 주 사양: 로그 (수출 있는 달만)
    a["y_asinh"] = np.arcsinh(a["exp"] / 1000)              # 보조: 0 포함 asinh (천 달러)
    a["rh"], a["rt"] = a["r"] + "|" + a["hs"], a["r"] + "|" + a["ym"].astype(str)
    tabs = {}
    for g in ("ALL", "VEH", "PARTS", "ALL_asinh"):
        d = a if g.startswith("ALL") else a[a["group"] == g]
        yv = "y_asinh" if g.endswith("asinh") else "y"
        d = d.dropna(subset=[yv]).copy()
        if d["hs"].nunique() < 4:
            continue
        cols = event_dummies(d, "s_us", "A_")
        t, n = run_es(d, yv, cols, ["rh", "rt"], "hs")
        t["group"], t["n"] = g, n
        tabs[g] = t
    es_a = pd.concat(tabs.values())
    es_a.to_csv(DST / "es_product.csv", index=False, encoding="utf-8-sig")
    plot_es({k: v for k, v in tabs.items() if k != "ALL_asinh"}, "A: product-level (within region-month), coefficient on US share", DST / "es_product.png")

    # ---- 설계 B: 시군구×월 ----
    w = sg[sg["ym"].dt.year == 2024].groupby(["r", "hs"])["exp"].sum().reset_index()
    w = w.merge(s[["s_us"]], left_on="hs", right_index=True)
    w["wx"] = w["exp"] * w["s_us"]
    E = w.groupby("r").agg(exp2024=("exp", "sum"), us2024=("wx", "sum"))
    E["E"] = E["us2024"] / E["exp2024"]
    E = E[E["exp2024"] > 5e6].sort_values("us2024", ascending=False)   # 2024년 500만 달러 이상
    E.to_csv(DST / "sgg_exposure.csv", encoding="utf-8-sig")
    b = sg.groupby(["r", "ym"], as_index=False)["exp"].sum()
    b = b[b["r"].isin(E.index)]
    full = pd.MultiIndex.from_product([E.index, sorted(sg["ym"].unique())], names=["r", "ym"])
    b = b.set_index(["r", "ym"])["exp"].reindex(full, fill_value=0).reset_index()
    b["sido"] = b["r"].str.split().str[0]
    b["E"] = b["r"].map(E["E"])
    b["y"] = np.log(b["exp"].where(b["exp"] > 0))
    b = b.dropna(subset=["y"])
    b["st"] = b["sido"] + "|" + b["ym"].astype(str)
    cols = event_dummies(b, "E", "B_")
    tb, nb = run_es(b, "y", cols, ["r", "st"], "r")
    tb.to_csv(DST / "es_region.csv", index=False, encoding="utf-8-sig")
    plot_es({"region exposure E_r": tb}, "B: sgg-level auto-chain exports, coefficient on exposure", DST / "es_region.png")
    plot_us(us, DST / "us_exports_monthly.png")

    # ---- 보고서 ----
    def summ(t, lab):
        pre = t[(t["k"] < -1) & (t["k"] >= -12)]
        post = t[t["k"] >= 0]
        return (f"| {lab} | {pre['b'].mean():.3f} | {(pre['b'].abs() > 1.96 * pre['se']).mean():.0%} | "
                f"{post[post['k'] <= 5]['b'].mean():.3f} | {post[post['k'] > 5]['b'].mean():.3f} |")
    lines = ["# 2025 미국 자동차 관세: 노출도와 이벤트 스터디 (1차)", "",
             f"- 기준월 2025-03 (k=-1), 창 {KMIN}~+{KMAX}개월, 양 끝 묶음",
             f"- 설계 A: 지역·품목 {a['rh'].nunique():,}개 × {a['ym'].nunique()}개월, 품목 {a['hs'].nunique()}개 (군집: HS6)",
             f"- 설계 B: 시군구 {len(E)}개 (군집: 시군구), 시도×월 FE", "",
             "| 사양 | 사전 평균 β (k=-12~-2) | 사전 유의 비율 | 사후 β (0~5개월) | 사후 β (6개월~) |",
             "|---|---|---|---|---|"]
    for g, t in tabs.items():
        lines.append(summ(t, f"A {g}" + (" (0 포함 asinh)" if g.endswith("asinh") else " (로그)")))
    lines.append(summ(tb, "B 지역 노출도"))
    lines += ["", "## 노출도 상위 시군구 (2024)", "",
              "| 시군구 | 자동차 공급망 수출(백만 달러) | 대미 추정(백만 달러) | 노출도 E |", "|---|---|---|---|"]
    for r, x in E.head(15).iterrows():
        lines.append(f"| {r} | {x.exp2024 / 1e6:,.0f} | {x.us2024 / 1e6:,.0f} | {x.E:.2f} |")
    lines += ["", "읽는 법: β는 대미 비중(또는 노출도) 0→1일 때 로그 수출의 차이 (예: -0.3 ≈ 26% 감소).",
              "사전 β가 0 근처이고 사후 β가 음수면 관세 효과. 완성차(VEH)와 부품(PARTS)의 하락 시점이",
              "4월(발효)보다 앞서면 발표·불확실성 효과(H1), 같거나 늦으면 관세 수준 효과."]
    (DST / "tariff_report.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    print(f"\n그림: {DST}/es_product.png, es_region.png, us_exports_monthly.png")


if __name__ == "__main__":
    main()
