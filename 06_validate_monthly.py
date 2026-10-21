"""
06. 월별 검증: 위성 지표(공장·산단 단위)를 시도 × 업종(KSIC 중분류)으로 묶어
    통계청 '시도/산업별 광공업생산지수'(월별, 물량 기준)와 비교

입력
  data/raw/eo/eo_<지표>_month.csv          04 단계 출력 (기본: ntl_vnp46, ntl_monthly)
  data/output/plant_unit_map.csv           03 단계: 공장 → 분석 단위
  data/interim/factories_geocoded.csv      02 단계: 공장 주소·업종·종업원
  data/raw/kosis/regional_ip.csv           00c 단계: KOSIS DT_1F02001
출력
  data/output/monthly_panel_sido_ind.csv   시도 × 업종 × 월 패널 (위성 지표 + 생산·출하·재고지수)
  data/output/monthly_validation.md        검증 결과
  data/output/kosis_industry_map.csv       KOSIS 산업 ↔ KSIC 대응표 (확인용)

방법
  셀(시도 s, 업종 k)의 위성 지표 = Σ_u w(u,s,k)·X_u,t / Σ w   (w = 단위 u 안에서 s·k에 속한 공장의 종업원 수)
  - 대형공장 단위에 속한 공장은 그 단위만, 나머지는 산단 단위로 배정 (중복 방지)
  - 산단 밖 소규모 개별입지 공장은 위성 단위가 없어 제외 → 셀별 '포괄률'을 함께 보고
  추정: ln IP_skt = β ln X_skt + 셀FE + 월FE  /  12개월 차분 ;  셀 단위 군집 표준오차
"""
import re
import sys
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from config import RAW, INT, OUT

EO_DEFAULT = ["ntl_vnp46", "ntl_monthly"]
EO_COL = {"ntl_vnp46": "mean", "ntl_monthly": "avg_rad_mean", "no2": "mean", "ntl_daily": "mean"}
MIN_COVER_EMP = 500          # 셀에 위성 단위로 잡힌 종업원이 이 이상일 때만 분석
SIDO = ["서울", "부산", "대구", "인천", "광주", "대전", "울산", "세종", "경기", "강원",
        "충북", "충남", "전북", "전남", "경북", "경남", "제주"]
GWANGJU_GU = {"광산구", "북구", "서구", "남구", "동구"}

# KSIC 10차 중분류 이름 (KOSIS 산업명 → 코드 대응용)
KSIC2_NAMES = {
    10: "식료품", 11: "음료", 12: "담배", 13: "섬유제품", 14: "의복", 15: "가죽",
    16: "목재", 17: "펄프", 18: "인쇄", 19: "코크스", 20: "화학물질", 21: "의료용 물질",
    22: "고무", 23: "비금속", 24: "1차 금속", 25: "금속가공", 26: "전자부품", 27: "의료, 정밀",
    28: "전기장비", 29: "기타 기계", 30: "자동차", 31: "기타 운송", 32: "가구", 33: "기타 제품",
}


def sido_of(addr):
    """주소 첫 단어 → 시도 약칭. '전남광주통합특별시'는 구(區) 이름으로 광주/전남 구분."""
    if not isinstance(addr, str) or not addr.strip():
        return None
    t = addr.split()
    a = t[0]
    if "광주" in a and "전남" in a or a.startswith("전남광주"):
        return "광주" if len(t) > 1 and t[1] in GWANGJU_GU else "전남"
    rules = [("충청북", "충북"), ("충청남", "충남"), ("전라북", "전북"), ("전북", "전북"), ("전라남", "전남"),
             ("경상북", "경북"), ("경상남", "경남"), ("강원", "강원"), ("제주", "제주"), ("세종", "세종")]
    for k, v in rules:
        if a.startswith(k):
            return v
    for s in SIDO:
        if a.startswith(s):
            return s
    return None


def kosis_sido(name):
    if not isinstance(name, str):
        return None
    if name.startswith("전국"):
        return "전국"
    return sido_of(name)


def kosis_ksic2(code, name):
    """KOSIS 산업 코드/이름 → KSIC 중분류(2자리). 대분류·소분류는 None."""
    c = str(code).strip()
    m = re.fullmatch(r"C?(\d{2})", c)
    if m and 10 <= int(m.group(1)) <= 33:
        return int(m.group(1))
    n = str(name)
    if any(x in n for x in ("총지수", "광업", "제조업", "전기", "가스")) and not any(
            x in n for x in ("전기장비",)):
        return None
    hits = [k for k, v in KSIC2_NAMES.items() if n.replace(" ", "").startswith(v.replace(" ", ""))]
    return hits[0] if len(hits) == 1 else None


def load_eo(names):
    out = None
    for n in names:
        f = RAW / "eo" / f"eo_{n}_month.csv"
        if not f.exists():
            print(f"(없음, 건너뜀) {f}")
            continue
        d = pd.read_csv(f)
        col = EO_COL.get(n, "mean")
        d["ym"] = pd.to_datetime(d["period_start"]).dt.to_period("M").dt.to_timestamp()
        d = d[["unit_id", "ym", col]].rename(columns={col: n})
        out = d if out is None else out.merge(d, on=["unit_id", "ym"], how="outer")
        print(f"위성 지표 {n}: {len(d):,}행, 단위 {d['unit_id'].nunique():,}개, {d['ym'].min():%Y-%m}~{d['ym'].max():%Y-%m}")
    if out is None:
        raise SystemExit("data/raw/eo/ 에 eo_ntl_vnp46_month.csv 등 월별 위성 파일이 없습니다.")
    return out


def weights():
    pm = pd.read_csv(OUT / "plant_unit_map.csv")
    fac = pd.read_csv(INT / "factories_geocoded.csv", low_memory=False,
                      usecols=lambda c: c in ("plant_id", "geo_addr", "addr_road", "ksic2", "employees"))
    fac["sido"] = fac["geo_addr"].map(sido_of)
    if "addr_road" in fac:
        fac["sido"] = fac["sido"].fillna(fac["addr_road"].map(sido_of))
    fac["emp"] = pd.to_numeric(fac.get("employees"), errors="coerce").fillna(1).clip(lower=1)
    # 대형공장 단위 우선, 없으면 산단 단위
    pm["pri"] = np.where(pm["unit_type"] == "plant", 0, 1)
    pm = pm.sort_values("pri").drop_duplicates("plant_id")
    w = fac.merge(pm[["plant_id", "unit_id"]], on="plant_id", how="left")
    tot = w.groupby(["sido", "ksic2"])["emp"].sum().rename("emp_all")
    w = w.dropna(subset=["unit_id", "sido", "ksic2"])
    w["ksic2"] = w["ksic2"].astype(int)
    uw = w.groupby(["unit_id", "sido", "ksic2"], as_index=False)["emp"].sum()
    cov = uw.groupby(["sido", "ksic2"])["emp"].sum().rename("emp_covered").to_frame().join(tot)
    cov["coverage"] = cov["emp_covered"] / cov["emp_all"]
    return uw, cov.reset_index()


def load_kosis():
    k = pd.read_csv(RAW / "kosis" / "regional_ip.csv", dtype={"C1": str, "C2": str, "ITM_ID": str})
    k["ym"] = pd.to_datetime(k["ym"])
    k["sido"] = k["C1_NM"].map(kosis_sido)
    k["ksic2"] = [kosis_ksic2(c, n) for c, n in zip(k["C2"], k["C2_NM"])]
    mp = k.drop_duplicates("C2")[["C2", "C2_NM", "ksic2"]].sort_values("C2")
    # 하위 산업(예: '자동차용 엔진')이 같은 중분류로 잡히면 코드가 가장 짧은(상위) 것만 사용
    mp["_len"] = mp["C2"].astype(str).str.len()
    keep = mp.dropna(subset=["ksic2"]).sort_values(["_len", "C2"]).drop_duplicates("ksic2")["C2"]
    mp["ksic2"] = mp["ksic2"].where(mp["C2"].isin(keep))
    k["ksic2"] = k["C2"].map(mp.set_index("C2")["ksic2"])
    mp = mp.drop(columns="_len")
    mp.to_csv(OUT / "kosis_industry_map.csv", index=False, encoding="utf-8-sig")
    item = {"생산지수(원지수)": "ip", "생산지수(계절조정)": "ip_sa", "생산자제품 출하지수(원지수)": "ship",
            "생산자제품 재고지수(원지수)": "inv"}
    k["var"] = k["ITM_NM"].map(item)
    k = k.dropna(subset=["var", "ksic2", "sido"])
    k = k[k["sido"] != "전국"]
    p = k.pivot_table(index=["sido", "ksic2", "ym"], columns="var", values="value", aggfunc="first").reset_index()
    print(f"KOSIS: 시도 {p['sido'].nunique()}개 × 업종 {p['ksic2'].nunique()}개 × {p['ym'].nunique()}개월 "
          f"(KSIC 대응 산업 {mp['ksic2'].notna().sum()}개 / 전체 {len(mp)}개 → {OUT / 'kosis_industry_map.csv'})")
    return p


def fe_reg(d, y, x, label):
    d = d.dropna(subset=[y, x])
    d = d[np.isfinite(d[y]) & np.isfinite(d[x])]
    if len(d) < 50 or d["cell"].nunique() < 3:
        return None
    f = f"{y} ~ {x} + C(cell) + C(ym_s)"
    m = smf.ols(f, d).fit(cov_type="cluster", cov_kwds={"groups": d["cell"].astype("category").cat.codes})
    base = smf.ols(f"{y} ~ C(cell) + C(ym_s)", d).fit()
    return dict(spec=label, b=m.params[x], se=m.bse[x], p=m.pvalues[x],
                r2w=1 - m.ssr / base.ssr, n=int(m.nobs), cells=d["cell"].nunique())


def main(eo_names):
    eo = load_eo(eo_names)
    uw, cov = weights()
    kp = load_kosis()

    m = uw.merge(eo, on="unit_id")
    rows = []
    for n in [c for c in eo_names if c in eo]:
        g = m.dropna(subset=[n]).copy()
        g[n] = g[n].where(g[n] > 0)
        g["wx"] = g["emp"] * np.log(g[n])
        a = g.dropna(subset=["wx"]).groupby(["sido", "ksic2", "ym"]).agg(wx=("wx", "sum"), w=("emp", "sum"))
        rows.append((a["wx"] / a["w"]).rename(f"ln_{n}"))
    sat = pd.concat(rows, axis=1).reset_index()

    df = kp.merge(sat, on=["sido", "ksic2", "ym"], how="inner").merge(cov, on=["sido", "ksic2"], how="left")
    df = df[df["emp_covered"] >= MIN_COVER_EMP].copy()
    df["cell"] = df["sido"] + "_" + df["ksic2"].astype(str)
    df["ym_s"] = df["ym"].dt.strftime("%Y-%m")
    for v in ("ip", "ip_sa", "ship", "inv"):
        if v in df:
            df[f"ln_{v}"] = np.log(df[v].where(df[v] > 0))
    df = df.sort_values(["cell", "ym"])
    # 12개월 차분: 행 순서가 아니라 날짜로 맞춤 (중간에 빠진 달이 있어도 정확)
    lncols = [c for c in df.columns if c.startswith("ln_")]
    lag = df[["cell", "ym"] + lncols].copy()
    lag["ym"] = lag["ym"] + pd.DateOffset(months=12)
    df = df.merge(lag, on=["cell", "ym"], how="left", suffixes=("", "_l12"))
    for c in lncols:
        df[f"d12_{c[3:]}"] = df[c] - df[f"{c}_l12"]
    df = df.drop(columns=[f"{c}_l12" for c in lncols])
    df.to_csv(OUT / "monthly_panel_sido_ind.csv", index=False, encoding="utf-8-sig")
    print(f"\n결합: {df['cell'].nunique()}개 셀(시도×업종), {len(df):,} 셀-월, "
          f"{df['ym'].min():%Y-%m}~{df['ym'].max():%Y-%m}, 종업원 포괄률 중앙값 {df.groupby('cell')['coverage'].first().median():.0%}")

    sats = [c[3:] for c in df.columns if c.startswith("ln_ntl") or c == "ln_no2"]
    res = []
    subsets = [("전체", df), ("자동차(30)", df[df["ksic2"] == 30]), ("기타운송·조선(31)", df[df["ksic2"] == 31]),
               ("1차금속(24)", df[df["ksic2"] == 24]), ("화학(20)", df[df["ksic2"] == 20]),
               ("전자(26)", df[df["ksic2"] == 26]), ("2025년 이후", df[df["ym"] >= "2025-01-01"])]
    for name, s in subsets:
        for x in sats:
            for y in ("ip", "ip_sa", "ship"):
                for pre, lab in (("ln_", "수준"), ("d12_", "12개월차분")):
                    if f"{pre}{y}" not in s:
                        continue
                    r = fe_reg(s, f"{pre}{y}", f"{pre}{x}", lab)
                    if r:
                        res.append(dict(표본=name, 위성=x, 결과=y, **r))
    rr = pd.DataFrame(res)
    rr.to_csv(OUT / "monthly_validation_results.csv", index=False, encoding="utf-8-sig")

    lines = ["# 월별 검증: 위성 지표 vs 시도/산업별 광공업생산지수", "",
             f"- 셀(시도×업종) {df['cell'].nunique()}개, {df['ym'].min():%Y-%m}~{df['ym'].max():%Y-%m}, "
             f"셀 FE + 월 FE, 셀 군집 표준오차", ""]
    for y in ("ip", "ship"):
        sub = rr[rr["결과"] == y]
        if sub.empty:
            continue
        lines += [f"## 결과: {'생산지수(원지수)' if y == 'ip' else '출하지수'}", "",
                  "| 표본 | 위성 | 사양 | β | (s.e.) | within R² | N | 셀 |", "|---|---|---|---|---|---|---|---|"]
        for r in sub.itertuples():
            st = "***" if r.p < .01 else "**" if r.p < .05 else "*" if r.p < .1 else ""
            lines.append(f"| {r.표본} | {r.위성} | {r.spec} | {r.b:.3f}{st} | ({r.se:.3f}) | {r.r2w:.3f} | {r.n} | {r.cells} |")
        lines.append("")
    # 자동차(30): 시도별로 위성 지표와 생산지수의 전년동월비 상관 (전체 기간 / 2025년 이후)
    if sats:
        x = f"d12_{sats[0]}"
        a = df[(df["ksic2"] == 30)].dropna(subset=[x, "d12_ip"])
        if len(a):
            lines += [f"## 자동차(30) 시도별: 전년동월비 상관 ({sats[0]} vs 생산지수)", "",
                      "| 시도 | 상관(전체) | 개월 | 상관(2025~) | 개월 | 종업원 포괄률 |", "|---|---|---|---|---|---|"]
            for sd, g in a.groupby("sido"):
                g2 = g[g["ym"] >= "2025-01-01"]
                c2 = g2[x].corr(g2["d12_ip"]) if len(g2) > 5 else np.nan
                lines.append(f"| {sd} | {g[x].corr(g['d12_ip']):.2f} | {len(g)} | {c2:.2f} | {len(g2)} | "
                             f"{g['coverage'].iloc[0]:.0%} |")
            lines.append("")
    (OUT / "monthly_validation.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main(sys.argv[1].split(",") if len(sys.argv) > 1 else EO_DEFAULT)
