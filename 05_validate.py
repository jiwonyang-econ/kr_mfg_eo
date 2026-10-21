"""
05. 위성 지표 검증: 산업단지 월별 생산·가동률을 위성 지표가 얼마나 설명하는가
입력: data/raw/eo/eo_<지표>_month.csv      (04 단계 GEE 출력)
      data/raw/complex_monthly.csv           (산단공 산업단지 동향)
      data/output/units_summary.csv          (03 단계)
출력: data/output/eo_panel_complex.csv       (단지×월 결합 패널)
      data/output/validation_results.csv
      data/output/validation_report.md

추정식 (단지 i, 월 t)
  [수준]   log Y_it  = β log X_it + α_i + δ_t + ε     α: 단지 FE, δ: 연월 FE
  [증가율] Δ12 log Y_it = β Δ12 log X_it + δ_t + ε  (전년동월 대비; 충격 탐지에 더 중요)
  표준오차는 단지 단위 군집. 업종 그룹별로도 따로 추정.
"""
import re
import numpy as np
import pandas as pd
import statsmodels.api as sm
from config import RAW, OUT, COMPLEX_PANEL_FILE, COMPLEX_PANEL_COLS
from utils import read_table, map_columns, to_num, norm_complex_name

# 지표별로 쓸 컬럼 (GEE 출력 속성명)
EO_MAIN = {"ntl_monthly": "avg_rad_mean", "ntl_daily": "mean", "no2": "mean", "lst": "mean"}
OUTCOMES = ["production", "exports", "utilization", "employment"]
MIN_OBS_GROUP = 60


def parse_ym(s):
    """'2021-03', '202103', '2021.03', '2021년 3월' → Timestamp(2021-03-01)."""
    m = re.search(r"(\d{4})\D*(\d{1,2})", str(s))
    return pd.Timestamp(int(m.group(1)), int(m.group(2)), 1) if m else pd.NaT


def load_eo():
    frames = []
    for name, col in EO_MAIN.items():
        f = RAW / "eo" / f"eo_{name}_month.csv"
        if not f.exists():
            continue
        d = pd.read_csv(f)
        if col not in d:
            raise KeyError(f"{f.name}에 '{col}' 컬럼이 없음: {list(d.columns)}")
        d = d[d["unit_type"] == "complex"]
        d["ym"] = pd.to_datetime(d["period_start"]).dt.to_period("M").dt.to_timestamp()
        frames.append(d[["unit_id", "ym", col]].rename(columns={col: name}))
        print(f"위성 지표 로드: {name} ({len(d):,}행)")
    if not frames:
        raise SystemExit(f"{RAW / 'eo'}에 eo_*_month.csv가 없습니다. 04 단계 출력을 넣어주세요.")
    out = frames[0]
    for f in frames[1:]:
        out = out.merge(f, on=["unit_id", "ym"], how="outer")
    return out


def load_complex_panel():
    p = map_columns(read_table(COMPLEX_PANEL_FILE), COMPLEX_PANEL_COLS,
                    required=["complex_name", "ym"], label="산단 동향")
    p["ym"] = p["ym"].map(parse_ym)
    for c in OUTCOMES + ["n_operating"]:
        if c in p:
            p[c] = to_num(p[c])
    p["complex_key"] = p["complex_name"].map(norm_complex_name)
    agg = {c: "sum" for c in ("production", "exports", "employment", "n_operating") if c in p}
    if "utilization" in p:
        agg["utilization"] = "mean"
    return p.groupby(["complex_key", "ym"], as_index=False).agg(agg)


def demean_2way(df, cols, i="unit_id", t="ym", iters=50, tol=1e-10):
    """교대 투영으로 단지·연월 고정효과 제거 (불균형 패널 대응)."""
    x = df[cols].astype(float).copy()
    for _ in range(iters):
        prev = x.values.copy()
        x = x - x.groupby(df[i]).transform("mean")
        x = x - x.groupby(df[t]).transform("mean")
        if np.nanmax(np.abs(x.values - prev)) < tol:
            break
    return x


def fe_ols(d, y, xs, fe_unit=True):
    d = d.dropna(subset=[y] + xs)
    if len(d) < MIN_OBS_GROUP or d["unit_id"].nunique() < 3:
        return None
    if fe_unit:
        z = demean_2way(d, [y] + xs)
    else:   # 연월 FE만
        z = d[[y] + xs] - d[[y] + xs].groupby(d["ym"]).transform("mean")
    m = sm.OLS(z[y], z[xs]).fit(cov_type="cluster", cov_kwds={"groups": d["unit_id"].astype("category").cat.codes})
    return dict(nobs=int(m.nobs), n_units=d["unit_id"].nunique(), r2_within=m.rsquared,
                **{f"b_{x}": m.params[x] for x in xs}, **{f"se_{x}": m.bse[x] for x in xs},
                **{f"p_{x}": m.pvalues[x] for x in xs})


def main():
    eo = load_eo()
    eo_vars = [c for c in EO_MAIN if c in eo]
    units = pd.read_csv(OUT / "units_summary.csv")
    panel = load_complex_panel()
    # 분기 자료면(모든 달이 3·6·9·12월) 위성 지표도 분기 평균으로 맞춤
    quarterly = panel["ym"].dt.month.isin([3, 6, 9, 12]).all()
    if quarterly:
        eo = eo.copy()
        eo["ym"] = eo["ym"].dt.to_period("Q").dt.end_time.dt.to_period("M").dt.to_timestamp()
        eo = eo.groupby(["unit_id", "ym"], as_index=False)[eo_vars].mean()
        print("산단 자료가 분기 단위 → 위성 지표를 분기 평균으로 집계")
    # 동향 자료의 단지명을 경계 이름 또는 별칭(경계 안 공장들이 적은 단지명)으로 연결
    units["key_raw"] = units["complex_raw"].map(norm_complex_name)
    units["key_alias"] = units.get("complex_alias", pd.Series(index=units.index, dtype=str)).map(
        lambda x: norm_complex_name(x) if isinstance(x, str) else None)
    keys = set(panel["complex_key"])
    units["complex_key"] = np.where(units["key_alias"].isin(keys), units["key_alias"], units["key_raw"])
    # 한 동향 단지에 경계가 여럿 붙는 경우(예: 구미 1단지·2·3단지·확장) → 위성 지표를 면적가중 평균해 단지 하나로
    w = units.set_index("unit_id")["area_km2"] if "area_km2" in units else None
    e = eo.merge(units[["unit_id", "complex_key"]], on="unit_id")
    e["_w"] = e["unit_id"].map(w).fillna(1.0) if w is not None else 1.0
    agg = {v: (lambda d, v=v: np.average(d[v].dropna(), weights=d.loc[d[v].notna(), "_w"])
                if d[v].notna().any() else np.nan) for v in eo_vars}
    e = e.groupby(["complex_key", "ym"]).apply(lambda d: pd.Series({v: f(d) for v, f in agg.items()}),
                                               include_groups=False).reset_index()
    meta = units.sort_values("employees", ascending=False).drop_duplicates("complex_key")
    e["unit_id"] = e["complex_key"]          # 이후 단위 = 동향 단지
    df = e.merge(meta[["complex_key", "dominant_group", "ind_diversity"]], on="complex_key") \
          .merge(panel, on=["complex_key", "ym"], how="inner")
    matched = df["unit_id"].nunique()
    print(f"\n결합: 단지 {matched:,}개 매칭 (위성 단지 {units['unit_id'].nunique():,}, "
          f"동향 단지 {panel['complex_key'].nunique():,}) | {len(df):,} 단지-월")
    unmatched = sorted(set(panel["complex_key"]) - set(units["complex_key"]))
    if unmatched:
        print(f"동향 자료에만 있는 단지 {len(unmatched)}개 (예: {unmatched[:8]}) → 이름 표기 확인")

    df = df.sort_values(["unit_id", "ym"])
    for c in eo_vars + [o for o in OUTCOMES if o in df]:
        v = df[c].where(df[c] > 0)
        df[f"ln_{c}"] = np.log(v)
        lag = 4 if quarterly else 12     # 전년 같은 기간
        df[f"d12_{c}"] = df[f"ln_{c}"] - df.groupby("unit_id")[f"ln_{c}"].shift(lag)
        # 12개월 전 관측이 실제 12개월 전인지 확인 (결측 월이 있으면 무효)
        gap = (df["ym"] - df.groupby("unit_id")["ym"].shift(lag)).dt.days
        df.loc[~gap.between(360, 370), f"d12_{c}"] = np.nan
    df.to_csv(OUT / "eo_panel_complex.csv", index=False, encoding="utf-8-sig")

    rows = []
    outcomes = [o for o in OUTCOMES if o in df]
    groups = [("전체", df)] + [(g, d) for g, d in df.groupby("dominant_group")]
    for y in outcomes:
        for spec, prefix, fe_unit in (("수준(단지·연월FE)", "ln_", True), ("전년동기증가율(기간FE)", "d12_", False)):
            # 지표 하나씩 + 전부 함께
            sets = [[v] for v in eo_vars] + ([eo_vars] if len(eo_vars) > 1 else [])
            for xs in sets:
                for g, d in groups:
                    r = fe_ols(d, prefix + y, [prefix + v for v in xs], fe_unit)
                    if r:
                        rows.append(dict(outcome=y, spec=spec, regressors="+".join(xs), group=g, **r))
    res = pd.DataFrame(rows)
    res.to_csv(OUT / "validation_results.csv", index=False, encoding="utf-8-sig")
    write_report(res, df, eo_vars, matched)
    print(f"\n저장: {OUT / 'validation_results.csv'}, {OUT / 'validation_report.md'}")


def write_report(res, df, eo_vars, matched):
    lines = ["# 위성 지표 검증 결과", "",
             f"- 매칭 단지 {matched}개, 기간 {df['ym'].min():%Y-%m} ~ {df['ym'].max():%Y-%m}",
             f"- 위성 지표: {', '.join(eo_vars)}", ""]
    for y in res["outcome"].unique():
        lines += [f"## 결과변수: {y}", ""]
        for spec in res["spec"].unique():
            sub = res[(res.outcome == y) & (res.spec == spec) & (res.regressors.str.count(r"\+") == 0)]
            if sub.empty:
                continue
            lines += [f"### {spec}", "", "| 업종그룹 | 지표 | β | (s.e.) | within R² | N | 단지 |", "|---|---|---|---|---|---|---|"]
            for r in sub.itertuples():
                x = [k for k in res.columns if k.startswith("b_") and pd.notna(getattr(r, k))][0][2:]
                b, se, p = getattr(r, "b_" + x), getattr(r, "se_" + x), getattr(r, "p_" + x)
                star = "***" if p < .01 else "**" if p < .05 else "*" if p < .1 else ""
                lines.append(f"| {r.group} | {r.regressors} | {b:.3f}{star} | ({se:.3f}) | "
                             f"{r.r2_within:.3f} | {r.nobs} | {r.n_units} |")
            lines.append("")
    lines += ["읽는 법: 증가율 사양의 β와 within R²가 업종 그룹별로 유의하게 나오는 지표가",
              "충격 분석(H1~H3)에 쓸 수 있는 지표입니다. R²가 낮은 업종은 본 분석에서 제외하거나",
              "보조 지표(야적장·항만 등)를 추가해야 합니다."]
    (OUT / "validation_report.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
