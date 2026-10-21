"""
01. 등록공장 원자료 정제
입력: 등록공장 자료 (팩토리온 파일, 표준데이터, 또는 00_fetch_api.py 결과 CSV)
      python 01_clean_factories.py data/raw/api/lndpcl_20261018.csv
출력: data/interim/factories_clean.csv

- 업종코드가 있으면 제조업(KSIC 10~34)만 남김 (필지정보 API에는 업종이 없어 이 단계 생략, 경고 출력)
- 숫자 컬럼 정리, 중복 제거
- 지오코딩용 주소 정리: '(동, 건물명)' 꼬리와 ', 302호' 같은 호수 제거
- 지식산업센터 등 한 건물에 여러 공장이 입주한 '공유주소' 표시
- 대형공장 플래그: 종업원·용지면적·건축면적 중 하나라도 기준 이상
"""
import re
import sys
import pandas as pd
from config import (FACTORY_FILE, FACTORY_COLS, FACTORY_REQUIRED, INT, MFG_KSIC2,
                    LARGE_PLANT_MIN_EMP, LARGE_PLANT_MIN_LAND_M2, LARGE_PLANT_MIN_BLDG_M2)
from utils import read_table, map_columns, to_num, ksic2_of, ksic_group


def clean_road_addr(a):
    """지오코딩용 주소 정리.
    '경기도 안양시 만안구 전파로 30,  유천팩토피아 222호 (안양동)' → '경기도 안양시 만안구 전파로 30'
    '경기도 안산시 단원구 성곡로 79, 636-3번지 608블럭' → '경기도 안산시 단원구 성곡로 79'  (도로명 우선)
    '전남광주통합특별시 광산구 하남동 500-4,5번지'       → '전남광주통합특별시 광산구 하남동 500-4'  (지번)
    """
    if not isinstance(a, str):
        return ""
    a = re.sub(r"\s+", " ", a.strip())
    # 1) 도로명주소: 처음 나오는 '…로/길 + 건물번호'까지
    m = re.match(r"^(.*?(?:로|길)\s*\d*(?:번길)?\s+\d+(?:-\d+)?)(?=$|[\s,(])", a)
    if m and "번지" not in m.group(1):
        return m.group(1)
    # 2) 지번주소: '…동/리 500-4,5번지 외 2필지' → '…동/리 500-4' (0번지는 의미 없는 값이라 버림)
    m = re.match(r"^(.*?[동리가읍면]\s*(?:산\s*)?)([1-9]\d*(?:-\d+)?)\s*(?:,\s*[\d,\-\s]+)?\s*번지", a)
    if m:
        return (m.group(1) + m.group(2)).strip()
    # 3) 그 밖: 괄호·쉼표 앞까지
    a = a.split(" (")[0]
    return a.split(",")[0].strip()


# 지식산업센터·아파트형공장을 지은 신탁사·시행사: 건물 전체 면적이 기재되어 대형공장처럼 보이지만 제조공장이 아님
DEVELOPER_PAT = (r"신탁|자산운용|자산개발|부동산|리츠|피에[프브]{1,2}이|PFV|프로젝트금융|디벨롭|"
                 r"개발\(주\)|\(주\)\S*개발$|건설(?!기계)|컨소시엄|정비사업|조합$")


def load_registry(path, prdctn=None):
    """필지정보(주소·면적)를 기본으로, 생산정보(업종·종업원·생산품·등록일)를 공장관리번호로 붙임."""
    df = map_columns(read_table(path), FACTORY_COLS, FACTORY_REQUIRED, label="등록공장")
    if prdctn is None:
        return df
    pr = map_columns(read_table(prdctn), FACTORY_COLS, ["factory_no"], label="생산정보")
    add = [c for c in ("ksic", "ksic_name", "ksic_all", "product", "employees", "reg_date") if c in pr]
    pr = pr.drop_duplicates("factory_no")[["factory_no"] + add]
    df = df.drop(columns=[c for c in add if c in df]).merge(pr, on="factory_no", how="left")
    hit = df["ksic"].notna().mean() if "ksic" in df else 0
    print(f"생산정보 결합: {df['factory_no'].isin(pr['factory_no']).sum():,} / {len(df):,}건 매칭 "
          f"(업종코드 보유 {hit:.1%}), 생산정보에만 있는 공장 {(~pr['factory_no'].isin(df['factory_no'])).sum():,}건")
    return df


def main(path=FACTORY_FILE, prdctn=None):
    df = load_registry(path, prdctn)
    n0 = len(df)

    for c in ("employees", "land_area", "bldg_area", "mfg_area"):
        if c in df:
            df[c] = to_num(df[c])

    if "ksic" in df:
        df["ksic2"] = df["ksic"].map(ksic2_of)
        no_code = df["ksic2"].isna()
        if no_code.any():
            print(f"업종코드 없는 공장 {no_code.sum():,}건은 일단 유지 (생산정보 미수집분이거나 미기재)")
        df = df[df["ksic2"].isin(MFG_KSIC2) | no_code].copy()
        df["ksic_group"] = df["ksic2"].map(ksic_group)
    else:
        print("경고: 업종코드 없음 → 제조업 필터 생략. 생산정보 API 자료와 공장관리번호로 합친 뒤 다시 실행하세요.")
        df["ksic2"], df["ksic_group"] = None, None
    n_mfg = len(df)

    # 지오코딩 주소: 도로명(정리본) 우선, 없으면 지번
    raw_road = df["addr_road"].fillna("") if "addr_road" in df else pd.Series("", index=df.index)
    road = raw_road.map(clean_road_addr)
    jibun = df["addr_jibun"].fillna("").str.strip() if "addr_jibun" in df else pd.Series("", index=df.index)
    df["geo_addr"] = road.where(road != "", jibun)
    df["geo_addr_type"] = (road != "").map({True: "road", False: "parcel"})
    df.loc[raw_road.str.contains("번지", na=False), "geo_addr_type"] = "parcel"   # 도로명 칸의 지번
    df.loc[df["geo_addr"] == "", ["geo_addr", "geo_addr_type"]] = [None, None]

    # 중복: 공장관리번호가 있으면 그것으로, 없으면 회사명+주소
    key = ["factory_no"] if "factory_no" in df else ["firm_name", "geo_addr"]
    df = df.drop_duplicates(subset=key).copy()

    # 같은 주소의 공장 수. 두 경우를 구분:
    #  - 다입주 건물(지식산업센터 등): 서로 다른 회사가 많음 → 건축면적이 건물 전체 면적이라 입주사 규모가 아님
    #  - 한 회사의 여러 등록(포스코·현대제철처럼 같은 부지에 공장 여러 건) → 대형공장 맞음, 03에서 한 단위로 묶음
    df["n_same_addr"] = df.groupby("geo_addr")["firm_name"].transform("size")
    df["shared_addr"] = df["n_same_addr"].fillna(1) > 1
    firm_key = df["firm_name"].fillna("").str.replace(r"\(주\)|주식회사|\s|\(.*?\)", "", regex=True).str[:4]
    n_firms = firm_key.groupby(df["geo_addr"]).transform("nunique")
    df["multi_tenant"] = n_firms.fillna(1) >= 5

    zero = pd.Series(0, index=df.index)
    land = df.get("land_area", zero)
    bldg = df.get("bldg_area", zero)
    # 500만㎡를 넘는 공장 건물은 사실상 없음 → 입력 오류(단위·연면적 혼동)로 표시.
    # 용지면적과의 비교는 쓰지 않음: 대형공장은 필지가 여러 개라 API의 용지면적이 일부 필지만 담는 경우가 많음
    # (예: 케이지모빌리티 용지 4.7만㎡ < 건축 28.7만㎡)
    df["area_suspect"] = bldg > 5_000_000
    if "employees" in df:
        # 건축면적 30만㎡ 이상인데 종업원 50명 미만 → 단위 입력 오류로 판단 (예: 용지 1,953㎡·건축 1,953,300㎡)
        df["area_suspect"] |= (bldg >= 300_000) & (df["employees"].fillna(-1).between(0, 49))
    b = bldg.where(~df["area_suspect"]).fillna(0)
    # 용지면적은 필지 전체라 대학 캠퍼스 안 소형 공장도 크게 잡힘 → 건축면적 5천㎡ 이상일 때만 인정
    df["large_plant"] = (df.get("employees", zero).fillna(0) >= LARGE_PLANT_MIN_EMP) | \
                        ((land.fillna(0) >= LARGE_PLANT_MIN_LAND_M2) & (b >= 5_000)) | \
                        (b >= LARGE_PLANT_MIN_BLDG_M2)
    df["developer"] = df["firm_name"].fillna("").str.contains(DEVELOPER_PAT, regex=True)
    df.loc[df["multi_tenant"] | df["developer"], "large_plant"] = False

    df.insert(0, "plant_id", [f"P{i:06d}" for i in range(len(df))])
    out = INT / "factories_clean.csv"
    df.to_csv(out, index=False, encoding="utf-8-sig")

    print(f"\n원자료 {n0:,}건 → 제조업 {n_mfg:,}건 → 중복 제거 후 {len(df):,}건")
    print(f"주소 없음: {df['geo_addr'].isna().sum():,}건 | 공유주소 공장: {df['shared_addr'].sum():,}건 "
          f"(그중 다입주 건물 {df['multi_tenant'].sum():,}건) | 면적 이상값: {df['area_suspect'].sum():,}건 "
          f"| 신탁·시행사: {df['developer'].sum():,}건")
    print(f"대형공장: {df['large_plant'].sum():,}건 (주소 기준 {df.loc[df['large_plant'], 'geo_addr'].nunique():,}개 부지)")
    for c in ("land_area", "bldg_area", "employees"):
        if c in df:
            print(f"  {c} 결측률 {df[c].isna().mean():.1%}")
    if "complex_name" in df:
        print(f"산업단지명 기재 공장: {df['complex_name'].notna().mean():.1%} "
              f"({df['complex_name'].nunique():,}개 단지)")
    if df["ksic_group"].notna().any():
        print("\n업종 그룹별:")
        print(df.groupby("ksic_group").agg(공장수=("plant_id", "size"), 대형=("large_plant", "sum")).to_string())
    print(f"\n저장: {out}")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("path", nargs="?", default=FACTORY_FILE, help="필지정보(lndpcl_*.csv) 또는 등록공장 파일")
    ap.add_argument("--prdctn", default=None, help="생산정보(prdctn_*.csv): 업종·종업원·생산품·등록일")
    a = ap.parse_args()
    main(a.path, a.prdctn)
