"""
00b. 한국산업단지공단 산업동향조사 통계 (분기) 수집 → 05 단계 검증용 단지별 패널
API: https://apis.data.go.kr/B550624/indparkstats/<기능>
     요청변수 serviceKey, srtStdrYm, endStdrYm, type=JSON (대문자)
특징: 기간을 넣으면 그 기간 '누적' 값을 단지별로 돌려줌(응답에 날짜 없음)
      → 분기마다 따로 호출 (시작=분기 첫 달, 종료=분기 마지막 달)

사용법 (Colab):
    import os; os.environ["DATA_GO_KR_KEY"] = "일반인증키"
    !python 00b_fetch_kicox.py --out /content/drive/MyDrive/kr_mfg_eo/api/kicox_stats
    !python 00b_fetch_kicox.py --test --out ...        # 2024년 1분기만
    !python 00b_fetch_kicox.py --start 2018Q1 --end 2026Q2 --out ...

출력:
    <out>/raw/<지표>_<분기>.json            원본 응답 (재실행 시 건너뜀)
    <out>/kicox_<지표>_long.csv             단지×분기×업종 (긴 형식)
    <out>/../complex_quarterly.csv           단지×분기 합계 패널 (05 단계 입력, data/raw에 복사해서 사용)
"""
import argparse
import json
import os
import time
from pathlib import Path
from urllib.parse import unquote

import pandas as pd
import requests

BASE = "https://apis.data.go.kr/B550624/indparkstats/"
OPS = {
    # '업종별' 기능이 실제로는 단지×업종 값을 돌려줌(irsttNm + induty01~12 + total) → 단지 패널로 충분.
    # '단지별(ByIrstt)' 기능은 명세에 없는 필수변수가 있는지 11 오류가 나서 쓰지 않음.
    # 지표: (기능명, 값의 의미)
    "production":  ("kicoxPrdRecByIndustryStatsService",     "생산실적"),
    "exports":     ("kicoxExportRecByIndustryStatsService",  "수출실적"),
    "utilization": ("kicoxOpRateByIndustryStatsService",     "가동률"),
    "employment":  ("kicoxEmpByIndustryStatsService",        "고용"),
    "n_operating": ("kicoxOpCmpnyByIndustryStatsService",    "가동업체"),
    "n_tenants":   ("kicoxMvnCmpnyByIndustryStatsService",   "입주업체"),
}
ID_COLS = {"irsttTyNm", "irsttNm", "indutyNm", "indutyCd"}


def quarters(start, end):
    s = pd.Period(start, freq="Q")
    e = pd.Period(end, freq="Q")
    return [pd.Period(ordinal=o, freq="Q") for o in range(s.ordinal, e.ordinal + 1)]


def ym_range(q):
    first = q.start_time.strftime("%Y%m")
    last = q.end_time.strftime("%Y%m")
    return first, last


def fetch(op, srt, end, key):
    for attempt in range(5):
        try:
            r = requests.get(BASE + op, params={"serviceKey": key, "srtStdrYm": srt,
                                                "endStdrYm": end, "type": "JSON"}, timeout=90)
            js = r.json()
            hdr = js.get("response", {}).get("header", {}) or js.get("OpenAPI_ServiceResponse", {}).get("cmmMsgHeader", {})
            code = hdr.get("resultCode") or hdr.get("returnReasonCode")
            if code == "00":
                return js
            msg = hdr.get("resultMsg") or hdr.get("returnAuthMsg")
            if code in ("22",):
                raise SystemExit(f"일일 호출 한도 초과 ({msg}). 내일 다시 실행하면 이어서 받습니다.")
            if code in ("03",):        # 데이터 없음
                return js
            if code in ("10", "11", "12", "20", "30", "31", "32"):
                return {"_error": f"{code} {msg}"}
            err = f"{code} {msg}"
        except (requests.RequestException, ValueError) as e:
            err = type(e).__name__
        time.sleep(10 * (attempt + 1))
    return {"_error": err}


def items_of(js):
    it = (js.get("response", {}).get("body", {}) or {}).get("items") or {}
    it = it.get("item", []) if isinstance(it, dict) else it
    return [it] if isinstance(it, dict) else (it or [])


def to_num(x):
    try:
        return float(str(x).replace(",", ""))
    except ValueError:
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/raw/kicox_stats")
    ap.add_argument("--start", default="2018Q1")
    ap.add_argument("--end", default=None, help="기본: 지난 분기")
    ap.add_argument("--only", default=",".join(OPS))
    ap.add_argument("--test", action="store_true")
    a = ap.parse_args()

    key = os.environ.get("DATA_GO_KR_KEY")
    if not key:
        raise SystemExit("환경변수 DATA_GO_KR_KEY에 인증키를 넣어주세요.")
    key = unquote(key)
    end = a.end or str(pd.Period(pd.Timestamp.today(), freq="Q") - 1)
    qs = [pd.Period("2024Q1", freq="Q")] if a.test else quarters(a.start, end)
    names = [n.strip() for n in a.only.split(",")]
    out = Path(a.out)
    (out / "raw").mkdir(parents=True, exist_ok=True)

    for name in names:
        op, label = OPS[name]
        recs, errs = [], []
        for q in qs:
            f = out / "raw" / f"{name}_{q}.json"
            if f.exists():
                js = json.loads(f.read_text(encoding="utf-8"))
            else:
                js = fetch(op, *ym_range(q), key)
                if "_error" in js:
                    errs.append(f"{q}:{js['_error']}")
                    continue
                f.write_text(json.dumps(js, ensure_ascii=False), encoding="utf-8")
                time.sleep(0.5)
            for it in items_of(js):
                base = {k: it.get(k) for k in ID_COLS if k in it}
                for k, v in it.items():
                    if k in ID_COLS:
                        continue
                    recs.append({**base, "quarter": str(q), "field": k, "value": to_num(v)})
        df = pd.DataFrame(recs)
        df.to_csv(out / f"kicox_{name}_long.csv", index=False, encoding="utf-8-sig")
        nq = df["quarter"].nunique() if len(df) else 0
        nu = df["irsttNm"].nunique() if "irsttNm" in df else 0
        print(f"[{name:12s}] {label}: {len(df):,}행, 분기 {nq}개, 단지 {nu}개"
              + (f" | 실패 {len(errs)}건 (예: {errs[:3]})" if errs else ""))
        if len(df) and a.test:
            print("   필드:", sorted(df["field"].unique())[:20])

    # 단지×분기 합계 패널 (05 단계 입력 형식)
    panel = None
    for name in ("production", "exports", "utilization", "employment"):
        f = out / f"kicox_{name}_long.csv"
        if not f.exists() or name not in names or f.stat().st_size < 10:
            continue
        d = pd.read_csv(f)
        if d.empty or "irsttNm" not in d:
            continue
        tot = d[d["field"].isin(["total", "totl", "sum", "tot"])]
        if tot.empty:     # 합계 필드가 없으면 업종 값 합산 (가동률은 평균)
            agg = "mean" if name == "utilization" else "sum"
            tot = d.groupby(["irsttNm", "quarter"], as_index=False)["value"].agg(agg)
        else:
            tot = tot.groupby(["irsttNm", "quarter"], as_index=False)["value"].first()
        tot = tot.rename(columns={"value": name})
        panel = tot if panel is None else panel.merge(tot, on=["irsttNm", "quarter"], how="outer")
    if panel is not None:
        panel["기준연월"] = panel["quarter"].map(lambda q: pd.Period(q, freq="Q").end_time.strftime("%Y%m"))
        panel = panel.rename(columns={"irsttNm": "단지명"})
        dst = out.parent / "complex_quarterly.csv"
        panel.to_csv(dst, index=False, encoding="utf-8-sig")
        print(f"\n단지×분기 패널: {len(panel):,}행, 단지 {panel['단지명'].nunique()}개 → {dst}")
        print(panel.head().to_string())


if __name__ == "__main__":
    main()
