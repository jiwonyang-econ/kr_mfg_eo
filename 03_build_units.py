"""
03. 위성 신호를 추출할 분석 단위(폴리곤) 구축
입력: data/interim/factories_geocoded.csv
      data/raw/complex_boundary.shp (산업단지 경계; 없으면 공장 좌표로 근사 경계 생성)
출력: data/output/units.geojson          (GEE 업로드용, EPSG:4326)
      data/output/plant_unit_map.csv      (공장 ↔ 단위 매핑, 기업 연결 단계에서 사용)
      data/output/units_summary.csv

단위 두 종류
  1) plant   : 대형공장 각각. 부지면적에서 반경을 역산한 원형 버퍼
  2) complex : 산업단지. 경계 폴리곤 + 입주 공장 업종 구성
"""
import re
import numpy as np
import pandas as pd
import geopandas as gpd
from config import INT, OUT, COMPLEX_SHP, COMPLEX_SHP_NAME_COLS
from utils import norm_complex_name

KOREA_CRS = "EPSG:5179"   # 미터 단위 투영 (UTM-K)


def plant_units(pts):
    """대형공장 단위. 같은 주소에 여러 건 등록된 대형공장(포스코처럼 한 부지에 공장 여러 건)은 한 단위로 묶음.
    반환: (단위 GeoDataFrame, 공장→단위 매핑)"""
    big = pts[pts["large_plant"]].copy()
    big["site_key"] = big["geo_addr"].fillna(big["plant_id"])
    zero = pd.Series(0.0, index=big.index)
    big["_bldg"] = big["bldg_area"].fillna(0) if "bldg_area" in big else zero
    big["_land"] = big["land_area"].fillna(0) if "land_area" in big else zero
    # 대표 공장 = 건축면적이 가장 큰 등록 건, 부지 면적 = 등록 건 합계
    rep = big.sort_values("_bldg", ascending=False).drop_duplicates("site_key").set_index("site_key")
    agg = big.groupby("site_key").agg(n_regs=("plant_id", "size"), bldg_sum=("_bldg", "sum"),
                                      land_sum=("_land", "sum"),
                                      employees=("employees", "sum") if "employees" in big else ("_bldg", "size"))
    u = rep.drop(columns=["employees"], errors="ignore").join(agg)
    # 부지 면적 추정: API 용지면적은 일부 필지만 담는 경우가 많아 건축면적×2와 비교해 큰 값 사용
    site_area = np.maximum(u["land_sum"], u["bldg_sum"] * 2)
    # 부지를 원으로 가정한 반경 + 100m 여유, 200m~2.5km
    u["radius_m"] = (np.sqrt(site_area / np.pi) + 100).clip(200, 2500)
    u = gpd.GeoDataFrame(u, geometry="geometry", crs=pts.crs)
    u["geometry"] = u.geometry.buffer(u["radius_m"])
    u["unit_id"] = "U_" + u["plant_id"]
    u["unit_type"] = "plant"
    u["unit_name"] = u["firm_name"]
    pmap = big[["plant_id", "site_key"]].merge(u[["unit_id"]], left_on="site_key", right_index=True)
    cols = ["unit_id", "unit_type", "unit_name", "ksic_group", "ksic2", "employees", "n_regs",
            "bldg_sum", "radius_m", "complex_name", "geometry"]
    return u.reset_index()[[c for c in cols if c in u.columns]], pmap[["plant_id", "unit_id"]]


def read_shp_korean(path):
    """국내 SHP는 .dbf가 cp949인 경우가 많음 → 한글이 깨지면 cp949로 다시 읽음."""
    g = gpd.read_file(path)
    cols = [c for c in g.columns if c != "geometry" and not pd.api.types.is_numeric_dtype(g[c])]
    txt = " ".join(map(str, g[cols].head(50).astype(str).values.ravel()[:500])) if cols else ""
    if re.search(r"[가-힣]", txt) is None and re.search(r"[À-ÿ]|\ufffd", txt):
        g = gpd.read_file(path, encoding="cp949")
    return g


def load_complex_polygons(pts):
    shp = COMPLEX_SHP if COMPLEX_SHP.exists() else next(iter(sorted(list(COMPLEX_SHP.parent.glob("*경계*.shp")) + list(COMPLEX_SHP.parent.glob("dam_dan*.shp")))), None)
    if shp is not None and shp.exists():
        cx = read_shp_korean(shp)
        if cx.crs is None:
            raise ValueError(f"{shp.name}에 좌표계(.prj)가 없습니다. 같은 이름의 .prj 파일을 함께 넣어주세요.")
        cx = cx.to_crs(KOREA_CRS)
        name_col = next((c for c in COMPLEX_SHP_NAME_COLS if c in cx.columns), None)
        if name_col is None:   # 이름 컬럼 자동 탐지: 값에 '산업단지/산단/농공단지'가 가장 많이 든 문자열 컬럼
            scores = {c: cx[c].astype(str).str.contains("단지|산단").mean()
                      for c in cx.columns if c != "geometry" and not pd.api.types.is_numeric_dtype(cx[c])}
            name_col = max(scores, key=scores.get) if scores and max(scores.values()) > 0.3 else None
            if name_col:
                print(f"단지명 컬럼 자동 선택: '{name_col}' (값 예: {cx[name_col].iloc[0]})")
        if name_col is None:
            raise KeyError(f"단지 경계 파일에서 단지명 컬럼을 못 찾음: {list(cx.columns)}\n"
                           f"config.py의 COMPLEX_SHP_NAME_COLS에 실제 컬럼명을 추가하세요.")
        id_col = next((c for c in ("DAN_ID", "단지코드", "ID") if c in cx.columns), None)
        type_col = next((c for c in ("DANJI_TYPE", "단지유형") if c in cx.columns), None)
        cx = cx.rename(columns={name_col: "complex_raw"})
        cx["complex_id"] = cx[id_col].astype(str) if id_col else cx["complex_raw"]
        # DANJI_TYPE: 1 국가, 2 일반, 3 도시첨단, 4 농공 (국토부 경계도면 기준)
        cx["complex_type"] = cx[type_col].astype(str).map({"1": "국가", "2": "일반", "3": "도시첨단", "4": "농공"}) \
            if type_col else None
        cx["geometry"] = cx.geometry.make_valid()
        # 같은 이름이 다른 지역에 여럿 있음(예: '고덕' 일반·농공) → 이름이 아니라 단지 ID로 조각을 합침
        cx = cx[["complex_id", "complex_raw", "complex_type", "geometry"]] \
            .dissolve(by="complex_id", as_index=False, aggfunc="first")
        cx["boundary_src"] = "shp"
        print(f"단지 경계: {len(cx):,}개 (원자료, 유형별 {cx['complex_type'].value_counts().to_dict()})")
        return cx
    # 경계 파일이 없을 때: 공장 등록자료의 단지명으로 묶어 볼록껍질+300m 버퍼 (근사)
    print("경고: 단지 경계 파일 없음 → 공장 좌표로 근사 경계를 만듭니다 (논문용으로는 원자료 경계 사용 권장)")
    g = pts.dropna(subset=["complex_name"])
    g = g[g["complex_name"].str.strip() != ""]
    hulls = g.groupby("complex_name").filter(lambda d: len(d) >= 5) \
             .dissolve(by="complex_name", as_index=False)
    hulls["geometry"] = hulls.geometry.convex_hull.buffer(300)
    hulls = hulls.rename(columns={"complex_name": "complex_raw"})[["complex_raw", "geometry"]]
    hulls["complex_id"] = hulls["complex_raw"]
    hulls["complex_type"] = None
    hulls["boundary_src"] = "approx_hull"
    return hulls


def complex_units(pts, cx):
    cx = cx.copy()
    cx["unit_id"] = [f"C{i:04d}" for i in range(len(cx))]
    cx["complex_key"] = cx["complex_raw"].map(norm_complex_name)
    j = gpd.sjoin(pts, cx[["unit_id", "geometry"]], how="inner", predicate="within")

    emp = j.pivot_table(index="unit_id", columns="ksic_group", values="employees",
                        aggfunc="sum", fill_value=0)
    share = emp.div(emp.sum(axis=1).replace(0, np.nan), axis=0)
    summ = pd.DataFrame({
        "n_plants": j.groupby("unit_id").size(),
        "employees": emp.sum(axis=1),
        "dominant_group": share.fillna(0).idxmax(axis=1).where(share.notna().any(axis=1)),
        "dominant_share": share.max(axis=1),
        # 업종 다양성 (1 - HHI): H4 지역 회복력 가설의 설명변수
        "ind_diversity": 1 - (share ** 2).sum(axis=1),
    })
    # 별칭: 경계 안 공장들이 등록자료에 적은 단지명 중 최빈값 (예: 경계 '반월특수(시화)' ↔ 등록 '시화국가산업단지')
    #  → 산단공 '산업단지 동향' 등 다른 자료와 이름으로 연결할 때 사용
    if "complex_name" in j:
        nm = j.dropna(subset=["complex_name"])
        summ["complex_alias"] = nm.groupby("unit_id")["complex_name"].agg(lambda s: s.value_counts().index[0])
        summ["alias_share"] = nm.groupby("unit_id")["complex_name"].agg(lambda s: s.value_counts(normalize=True).iloc[0])
    share.columns = [f"empshare_{c}" for c in share.columns]
    cx = cx.merge(summ, left_on="unit_id", right_index=True, how="left") \
           .merge(share, left_on="unit_id", right_index=True, how="left")
    # 실제 공장이 있는 부분만 남긴 '유효 경계': 경계도면에는 녹지·주거·호수까지 들어간 경우가 있음
    # (예: 반월특수(시화) 경계 약 135㎢) → 위성 신호가 희석되지 않도록 공장 좌표 250m 버퍼와의 교집합 사용
    buf = j.assign(geometry=j.geometry.buffer(250)).dissolve(by="unit_id")[["geometry"]]
    cx = cx.set_index("unit_id")
    cx["area_full_km2"] = cx.geometry.area / 1e6
    eff = cx.loc[buf.index].geometry.intersection(buf.geometry)
    use = cx.loc[buf.index, "n_plants"].fillna(0) >= 5          # 공장이 너무 적으면 원래 경계 유지
    cx.loc[use[use].index, "geometry"] = eff[use]
    cx["footprint"] = "full"
    cx.loc[use[use].index, "footprint"] = "plants_250m"
    cx = cx.reset_index()
    cx["unit_type"] = "complex"
    cx["unit_name"] = cx["complex_raw"]
    cx["ksic_group"] = cx["dominant_group"]
    cx["area_km2"] = cx.geometry.area / 1e6
    print(f"유효 경계 적용 단지 {int((cx['footprint'] == 'plants_250m').sum()):,}개: 면적 합 "
          f"{cx.loc[cx['n_plants'].fillna(0) > 0, 'area_full_km2'].sum():,.0f}㎢ → {cx.loc[cx['n_plants'].fillna(0) > 0, 'area_km2'].sum():,.0f}㎢ (공장 있는 단지 기준)")
    return cx, j[["plant_id", "unit_id"]]


def main():
    fac = pd.read_csv(INT / "factories_geocoded.csv")
    fac = fac.dropna(subset=["lon", "lat"])
    pts = gpd.GeoDataFrame(fac, geometry=gpd.points_from_xy(fac.lon, fac.lat),
                           crs="EPSG:4326").to_crs(KOREA_CRS)

    pu, pmap_p = plant_units(pts)
    cx = load_complex_polygons(pts)
    cu, pmap_c = complex_units(pts, cx)
    cu = cu[cu["n_plants"].fillna(0) > 0]      # 공장이 하나도 안 잡힌 경계 제외

    pmap = pd.concat([pmap_p, pmap_c], ignore_index=True)
    pmap["unit_type"] = np.where(pmap["unit_id"].str.startswith("U_"), "plant", "complex")
    pmap.to_csv(OUT / "plant_unit_map.csv", index=False, encoding="utf-8-sig")

    keep = ["unit_id", "unit_type", "unit_name", "ksic_group", "employees", "geometry"]
    units = pd.concat([pu[keep + ["radius_m"]], cu[keep + ["n_plants", "ind_diversity", "area_km2"]]],
                      ignore_index=True)
    units = gpd.GeoDataFrame(units, geometry="geometry", crs=KOREA_CRS)
    units["geometry"] = units.geometry.simplify(20)   # GEE 업로드 용량 줄이기 (20m 허용오차)
    units.to_crs("EPSG:4326").to_file(OUT / "units.geojson", driver="GeoJSON")

    cu.drop(columns="geometry").to_csv(OUT / "units_summary.csv", index=False, encoding="utf-8-sig")
    print(f"\n분석 단위: 대형공장 부지 {len(pu):,}개 (등록 {len(pmap_p):,}건) + 산업단지 {len(cu):,}개 = {len(units):,}개")
    print(f"산단에 매칭된 공장: {pmap_c['plant_id'].nunique():,} / {len(pts):,} "
          f"({pmap_c['plant_id'].nunique() / len(pts):.1%}) — 나머지는 개별입지")
    print("\n산단 주력업종 분포:")
    print(cu["dominant_group"].value_counts().to_string())
    print(f"\n저장: {OUT / 'units.geojson'}")


if __name__ == "__main__":
    main()
