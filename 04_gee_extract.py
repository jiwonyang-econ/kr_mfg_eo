"""
04. Google Earth Engine에서 분석 단위별 위성 지표 추출
입력: data/output/units.geojson
출력: Google Drive 'kr_mfg_eo' 폴더에 지표별 CSV (월별 또는 주별 패널)
      → 내려받아 data/raw/eo/ 에 넣으면 05 단계에서 자동 결합

사전 준비 (한 번만):
    pip install earthengine-api
    earthengine authenticate
    GEE 클라우드 프로젝트 생성 (학술·비영리 무료 등록: https://earthengine.google.com/noncommercial/)

사용법:
    python 04_gee_extract.py --project 내프로젝트ID --dry-run      # 작업 계획만 출력
    python 04_gee_extract.py --project 내프로젝트ID                # 월별, 전 지표
    python 04_gee_extract.py --project ID --freq week --only ntl_daily,no2   # 이벤트 스터디용 주별

지표
    ntl_monthly : VIIRS DNB 월별 합성 (NOAA VCMSLCFG, 미광 보정), avg_rad + 관측일수
    ntl_daily   : VIIRS VNP46A2 일별 (BRDF·달빛 보정, gap-filled) → 기간 평균
    ntl_vnp46   : VNP46A2 일별, gap-fill 없이 품질 양호(플래그 0·1) 관측만 → 기간 평균 + 유효 관측일 수 (권장)
    no2         : Sentinel-5P TROPOMI 대류권 NO2 (µmol/m²) → 기간 평균
    lst         : Landsat 8/9 지표면온도 (켈빈, 구름 마스크) → 기간 평균 (관측 희소)

주의: 데이터셋 ID·밴드명은 GEE 카탈로그 갱신으로 바뀔 수 있으니 실행 전 카탈로그에서 확인하세요.
"""
import argparse
import json
from datetime import date
import pandas as pd
from config import OUT, START, END

DATASETS = {
    "ntl_monthly": dict(id="NOAA/VIIRS/DNB/MONTHLY_V1/VCMSLCFG", bands=["avg_rad", "cf_cvg"],
                        scale=463.83, native_freq="month"),
    "ntl_daily":   dict(id="NASA/VIIRS/002/VNP46A2", bands=["Gap_Filled_DNB_BRDF_Corrected_NTL"],
                        scale=463.83, native_freq="day"),
    # 일별 VNP46A2에서 '실제 관측'만 사용: gap-filled(이전 값으로 메움) 대신 BRDF 보정 원값 +
    # 품질 플래그 0(고품질 지속)·1(고품질 일시)만 남김 → 월별·주별 평균과 유효 관측일 수(count)
    "ntl_vnp46":   dict(id="NASA/VIIRS/002/VNP46A2", bands=["DNB_BRDF_Corrected_NTL"],
                        scale=463.83, native_freq="day"),
    "no2":         dict(id="COPERNICUS/S5P/OFFL/L3_NO2", bands=["tropospheric_NO2_column_number_density"],
                        scale=1113.2, native_freq="day"),
    "lst":         dict(id=["LANDSAT/LC08/C02/T1_L2", "LANDSAT/LC09/C02/T1_L2"], bands=["ST_B10"],
                        scale=100, native_freq="16day"),
}


def periods(freq, start=None, end=None):
    start, end = start or START, end or END
    """[(시작일, 다음기간 시작일)] — GEE filterDate는 끝 날짜를 포함하지 않음."""
    rule = {"month": "MS", "week": "W-MON"}[freq]
    starts = list(pd.date_range(start, end, freq=rule))
    ends = starts[1:] + [pd.Timestamp(end) + pd.Timedelta(days=1)]
    return [(a.strftime("%Y-%m-%d"), b.strftime("%Y-%m-%d")) for a, b in zip(starts, ends)]


def output_columns(name):
    """mean+count 결합 리듀서의 출력 속성명: 밴드 1개면 mean/count, 여러 개면 <밴드>_mean/<밴드>_count."""
    bands = DATASETS[name]["bands"]
    if len(bands) == 1:
        return ["mean", "count"]
    return [f"{b}_{s}" for b in bands for s in ("mean", "count")]


def build_collection(ee, name):
    d = DATASETS[name]
    if name == "lst":
        def mask_scale(img):
            qa = img.select("QA_PIXEL")
            clear = qa.bitwiseAnd(1 << 3).eq(0).And(qa.bitwiseAnd(1 << 4).eq(0))
            return img.select("ST_B10").multiply(0.00341802).add(149.0) \
                      .updateMask(clear).copyProperties(img, ["system:time_start"])
        col = ee.ImageCollection(d["id"][0]).merge(ee.ImageCollection(d["id"][1])).map(mask_scale)
        return col
    col = ee.ImageCollection(d["id"]).select(d["bands"])
    if name == "no2":
        col = col.map(lambda i: i.multiply(1e6).copyProperties(i, ["system:time_start"]))  # mol/m² → µmol/m²
    if name == "ntl_vnp46":
        def qmask(img):
            q = img.select("Mandatory_Quality_Flag")
            v = img.select("DNB_BRDF_Corrected_NTL")
            return v.updateMask(q.lte(1).And(v.lt(65535))).copyProperties(img, ["system:time_start"])
        return ee.ImageCollection(d["id"]).map(qmask)
    if name == "ntl_daily":
        # VNP46A2 정수 저장값의 스케일 확인 필요: GEE 카탈로그의 scale 표기에 맞춰 조정
        col = col.map(lambda i: i.updateMask(i.lt(65535)).copyProperties(i, ["system:time_start"]))
    return col


def _tagger(period_start, n_img):
    """GEE map은 인자 1개짜리 함수만 받으므로 기간 정보를 클로저로 넘김."""
    def f(feat):
        return feat.set({"period_start": period_start, "n_images": n_img}).setGeometry(None)
    return f


def extract(ee, fc, name, freq, region_bbox, drive_folder, start=None, end=None, tag=""):
    d = DATASETS[name]
    col = build_collection(ee, name).filterBounds(region_bbox)
    feats = []
    for a, b in periods(freq, start, end):
        sub = col.filterDate(a, b)
        img = sub.mean()
        n_img = sub.size()
        stats = img.reduceRegions(
            collection=fc, reducer=ee.Reducer.mean().combine(ee.Reducer.count(), sharedInputs=True),
            scale=d["scale"], tileScale=4,
        ).map(_tagger(a, n_img))
        feats.append(stats)
    out = ee.FeatureCollection(feats).flatten()
    task = ee.batch.Export.table.toDrive(
        collection=out, description=f"kr_mfg_eo_{name}_{freq}{tag}",
        folder=drive_folder, fileNamePrefix=f"eo_{name}_{freq}{tag}", fileFormat="CSV",
        selectors=["unit_id", "unit_type", "period_start", "n_images"] + output_columns(name),
    )
    task.start()
    return task


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--project", required=True, help="GEE 클라우드 프로젝트 ID")
    ap.add_argument("--freq", choices=["month", "week"], default="month")
    ap.add_argument("--only", default=",".join(DATASETS), help="쉼표로 구분한 지표 목록")
    ap.add_argument("--asset", default=None,
                    help="단위 폴리곤을 GEE 자산으로 올렸다면 자산 경로 (단위 수천 개 이상이면 권장)")
    ap.add_argument("--drive-folder", default="kr_mfg_eo")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--start", default=None, help="시작일 (예: 2024-01-01). 생략 시 config.START")
    ap.add_argument("--end", default=None, help="종료일 (예: 2024-03-31). 생략 시 config.END")
    ap.add_argument("--tag", default="", help="출력 파일 이름 꼬리표 (예: _pilot)")
    ap.add_argument("--unit-filter", default=None,
                    help="units.geojson 속성으로 단위 제한 (예: 'ksic_group=자동차·조선'). 주별처럼 무거운 추출용")
    args = ap.parse_args()

    names = [n.strip() for n in args.only.split(",")]
    bad = [n for n in names if n not in DATASETS]
    if bad:
        raise SystemExit(f"알 수 없는 지표: {bad}. 가능: {list(DATASETS)}")

    gj = json.loads((OUT / "units.geojson").read_text(encoding="utf-8"))
    if args.unit_filter:
        k, v = args.unit_filter.split("=", 1)
        gj["features"] = [f for f in gj["features"] if str(f["properties"].get(k)) == v]
    n_units = len(gj["features"])
    per = periods(args.freq, args.start, args.end)
    print(f"단위 {n_units:,}개 × 기간 {len(per)}개 ({per[0][0]} ~ {per[-1][1]}) × 지표 {names}")
    if args.freq == "week" and "ntl_monthly" in names:
        print("경고: ntl_monthly는 월별 원자료라 주별 추출이 의미 없음 → 제외")
        names.remove("ntl_monthly")
    mb = (OUT / "units.geojson").stat().st_size / 1e6
    print(f"units.geojson 크기 {mb:.1f}MB")
    if (n_units > 3000 or mb > 8) and not args.asset:
        print("경고: 단위가 많으면 GeoJSON 직접 전송이 요청 한도를 넘을 수 있음 → "
              "units.geojson을 GEE 자산으로 업로드 후 --asset 사용 권장")
    if args.dry_run:
        print("dry-run: 작업을 제출하지 않았습니다.")
        return

    import ee
    ee.Initialize(project=args.project)
    if args.asset:
        fc = ee.FeatureCollection(args.asset)
    else:
        for f in gj["features"]:   # GEE로 보낼 속성만 남김
            f["properties"] = {k: f["properties"].get(k) for k in ("unit_id", "unit_type")}
        fc = ee.FeatureCollection(gj)
    bbox = ee.Geometry.Rectangle([124.5, 33.0, 131.0, 38.7])

    for n in names:
        t = extract(ee, fc, n, args.freq, bbox, args.drive_folder, args.start, args.end, args.tag)
        print(f"제출: {n} → task {t.id}  (진행 상황: https://code.earthengine.google.com/tasks)")
    print(f"\n완료되면 Google Drive '{args.drive_folder}' 폴더의 CSV를 data/raw/eo/ 에 넣으세요.")


if __name__ == "__main__":
    main()
