"""
파이프라인 테스트용 가짜 데이터 생성 (실제 수치 아님).
    python tests/make_sample_data.py raw   # 01~03 단계 입력 생성
    python tests/make_sample_data.py eo    # 03 실행 후, 가짜 GEE 출력 생성
"""
import sys
import hashlib
from pathlib import Path
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import RAW, OUT, START, END   # noqa: E402

rng = np.random.default_rng(0)
COMPLEXES = {  # 이름: (경도, 위도, 주력 KSIC2 목록)
    "반월국가산업단지": (126.80, 37.30, [25, 29, 26]), "시화국가산업단지": (126.73, 37.33, [25, 29, 20]),
    "울산미포국가산업단지": (129.35, 35.52, [30, 31, 20]), "구미국가산업단지": (128.40, 36.11, [26, 27]),
    "창원국가산업단지": (128.65, 35.21, [29, 28, 30]), "여수국가산업단지": (127.69, 34.84, [20, 19]),
    "포항철강산업단지": (129.38, 35.99, [24, 25]), "광양국가산업단지": (127.73, 34.93, [24, 19]),
    "남동국가산업단지": (126.70, 37.40, [25, 29, 22]), "아산테크노밸리": (127.00, 36.90, [26, 30]),
    "오창과학산업단지": (127.43, 36.71, [26, 21]), "녹산국가산업단지": (128.86, 35.09, [25, 29]),
}


def make_raw():
    """좌표는 지오코딩 캐시에 미리 넣어 둠 → 02 단계가 API 호출 없이 캐시로 처리."""
    rows, cache = [], []
    i = 0
    for name, (lon, lat, k2s) in COMPLEXES.items():
        for _ in range(rng.integers(80, 160)):
            k2 = rng.choice(k2s + [10, 22, 25]) if rng.random() < .25 else rng.choice(k2s)
            emp = int(rng.lognormal(3.2, 1.1))
            land = int(emp * rng.uniform(40, 120))
            addr = f"테스트도 테스트시 산단로 {i}"
            x, y = lon + rng.normal(0, .008), lat + rng.normal(0, .006)
            rows.append({"회사명": f"(주)테스트{i}", "단지명": name, "대표업종": f"C{k2}{rng.integers(100, 999)}",
                         "업종명": "테스트", "생산품": "테스트품", "공장대표주소(도로명)": addr,
                         "공장대표주소(지번)": f"테스트도 테스트시 {i}번지", "종업원합계": f"{emp:,}",
                         "용지면적": f"{land:,}", "건축면적": str(int(land * .5)), "등록구분": "등록"})
            cache.append({"geo_addr": addr, "lon": x, "lat": y, "geo_status": "road"})
            i += 1
    for _ in range(300):   # 개별입지 공장 + 비제조업 섞기
        k2 = int(rng.choice([10, 20, 25, 29, 46, 58]))
        addr = f"테스트도 개별입지로 {i}"
        rows.append({"회사명": f"(주)개별{i}", "단지명": "", "대표업종": f"{k2}111", "업종명": "테스트",
                     "생산품": "", "공장대표주소(도로명)": addr if rng.random() > .1 else "",
                     "공장대표주소(지번)": f"테스트도 개별입지 {i}번지", "종업원합계": str(int(rng.lognormal(2.5, 1))),
                     "용지면적": str(int(rng.uniform(500, 20000))), "건축면적": "", "등록구분": "등록"})
        cache.append({"geo_addr": addr, "lon": 126.5 + rng.random() * 3, "lat": 35 + rng.random() * 2.5,
                      "geo_status": "road"})
        i += 1
    # 대형공장 몇 개 (제철소·완성차·조선)
    for name, k2, emp, land in [("포항철강산업단지", 24, 9000, 9_000_000), ("울산미포국가산업단지", 30, 30000, 5_000_000),
                                ("울산미포국가산업단지", 31, 20000, 6_000_000), ("광양국가산업단지", 24, 6000, 15_000_000)]:
        lon, lat, _ = COMPLEXES[name]
        addr = f"테스트도 대형로 {i}"
        rows.append({"회사명": f"(주)대형{i}", "단지명": name, "대표업종": f"C{k2}111", "업종명": "테스트",
                     "생산품": "", "공장대표주소(도로명)": addr, "공장대표주소(지번)": "", "종업원합계": f"{emp:,}",
                     "용지면적": f"{land:,}", "건축면적": "", "등록구분": "등록"})
        cache.append({"geo_addr": addr, "lon": lon, "lat": lat, "geo_status": "road"})
        i += 1
    df = pd.DataFrame(rows)
    df.loc[len(df)] = df.iloc[0]   # 중복 1건
    df.to_csv(RAW / "factory_registry.csv", index=False, encoding="cp949")   # 공공데이터포털처럼 cp949

    from config import GEOCODE_CACHE
    pd.DataFrame(cache).to_csv(GEOCODE_CACHE, index=False, encoding="utf-8-sig")

    # 단지 경계 (원형 근사, 실제는 산업입지정보시스템 SHP)
    gdf = gpd.GeoDataFrame({"단지명": list(COMPLEXES)},
                           geometry=[Point(v[0], v[1]) for v in COMPLEXES.values()], crs="EPSG:4326")
    gdf = gdf.to_crs("EPSG:5179")
    gdf["geometry"] = gdf.buffer(2500)
    gdf.to_crs("EPSG:4326").to_file(RAW / "complex_boundary.shp", encoding="utf-8")

    # 산단 동향 (단지명 표기를 일부러 다르게) + 잠재 활동지수 저장
    months = pd.date_range(START, END, freq="MS")
    lat_rows, pan = [], []
    for name in list(COMPLEXES) + ["가상없는국가산업단지"]:
        level = rng.uniform(5000, 50000)
        a = np.cumsum(rng.normal(0, .03, len(months))) + 0.1 * np.sin(2 * np.pi * months.month / 12)
        # 2019-07 수출규제, 2025-04 관세 충격 흉내
        a = a - 0.12 * ((months >= "2019-07-01") & (months < "2020-03-01")) \
              - 0.15 * ((months >= "2025-04-01") & (months < "2025-10-01"))
        for m, v in zip(months, a):
            shown = name.replace("국가산업단지", "(국가)") if rng.random() < .5 else name
            pan.append({"단지명": shown, "기준연월": m.strftime("%Y%m"),
                        "생산(억원)": f"{level * np.exp(v + rng.normal(0, .04)):,.0f}",
                        "수출(백만불)": f"{level / 15 * np.exp(v + rng.normal(0, .06)):.1f}",
                        "가동률(%)": f"{np.clip(75 + 40 * v + rng.normal(0, 2), 30, 99):.1f}",
                        "고용(명)": str(int(level / 3 * np.exp(v / 4))), "가동업체": "100"})
            lat_rows.append({"complex_raw": name, "ym": m, "latent": v})
    pd.DataFrame(pan).to_csv(RAW / "complex_monthly.csv", index=False, encoding="cp949")
    pd.DataFrame(lat_rows).to_csv(RAW / "_latent_for_test.csv", index=False)
    print("가짜 원자료 생성 완료:", RAW)


def make_eo():
    """03 산출물의 unit_id를 써서, 잠재 활동지수와 상관된 가짜 GEE 출력 생성."""
    units = pd.read_csv(OUT / "units_summary.csv")
    lat = pd.read_csv(RAW / "_latent_for_test.csv", parse_dates=["ym"])
    d = units[["unit_id", "complex_raw", "dominant_group"]].merge(lat, on="complex_raw")
    # 업종별로 신호 강도를 다르게: 중화학은 강하고 전자는 약하게 (실제 예상 패턴 흉내)
    k = d["dominant_group"].map({"철강·금속": 1.0, "석유화학·화학": .9, "자동차·조선": .8,
                                 "기계": .6, "전자·반도체": .2}).fillna(.4)
    (RAW / "eo").mkdir(exist_ok=True)
    base = dict(unit_id=d["unit_id"], unit_type="complex", period_start=d["ym"].dt.strftime("%Y-%m-%d"), n_images=25)
    pd.DataFrame({**base, "avg_rad_mean": 30 * np.exp(k * d["latent"] + rng.normal(0, .08, len(d))),
                  "avg_rad_count": 400, "cf_cvg_mean": 12, "cf_cvg_count": 400}) \
        .to_csv(RAW / "eo" / "eo_ntl_monthly_month.csv", index=False)
    pd.DataFrame({**base, "mean": 80 * np.exp(k * d["latent"] + rng.normal(0, .15, len(d))), "count": 30}) \
        .to_csv(RAW / "eo" / "eo_no2_month.csv", index=False)
    print("가짜 GEE 출력 생성 완료:", RAW / "eo")


if __name__ == "__main__":
    {"raw": make_raw, "eo": make_eo}[sys.argv[1]]()
