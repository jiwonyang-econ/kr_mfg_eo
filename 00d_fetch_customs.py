"""
00d. 관세청 품목별 국가별 수출입실적 (공공데이터포털 15100475) → 대미 노출도·월별 수출
API: https://apis.data.go.kr/1220000/nitemtrade/getNitemtradeList
     serviceKey, strtYymm, endYymm (1년 이내), cntyCd (필수, 2자리), hsSgn (선택)
응답: year(연월), statCd, statCdCntnKor1, hsCd, statKor, expDlr, expWgt, impDlr, impWgt, balPayments

사용법 (Colab, 세션 시작 셀 이후):
    # 1) 응답 형태 확인 (HS 코드 자릿수, 행 수, 페이지 여부)
    !python 00d_fetch_customs.py peek --cnty US --ym 202501
    !python 00d_fetch_customs.py peek --cnty US --ym 202501 --hs 87

    # 2) 받기: 국가별 × 연도별 (연 단위로 나눠 요청, 결과는 캐시돼 끊겨도 이어받음)
    !python 00d_fetch_customs.py get --cnty US,CN,MX,VN,JP,DE --start 201801 --end 202608
    !python 00d_fetch_customs.py get --cnty US --start 201801 --end 202608 --hs 87,84,85,72,73,40

출력: data/raw/customs/raw/<국가>_<연도>_<hs>.xml   (원본, 캐시)
      data/raw/customs/nitem_trade.csv               (ym, cnty, hs, 품목명, 수출·수입 금액·중량)
"""
import argparse
import os
import time
import xml.etree.ElementTree as ET
from pathlib import Path
from urllib.parse import unquote

import pandas as pd
import requests

URL = "https://apis.data.go.kr/1220000/nitemtrade/getNitemtradeList"
OUT = Path("data/raw/customs")
# 자동차 공급망 HS4 (4자리를 넣으면 6자리 품목이 돌아옴)
AUTO_HS4 = ["8703", "8704", "8706", "8707", "8708", "8407", "8408", "8409", "4011", "8507", "8511",
            "8512", "8536", "8544", "9401", "8302", "8483", "7320", "7007", "7010"]
NUM = ["expDlr", "expWgt", "impDlr", "impWgt", "balPayments"]


def key():
    k = os.environ.get("DATA_GO_KR_KEY")
    if not k or len(k) < 30:
        raise SystemExit("환경변수 DATA_GO_KR_KEY에 실제 인증키를 넣어주세요.")
    return unquote(k)


def call(params, retries=5):
    err = None
    for i in range(retries):
        try:
            r = requests.get(URL, params=dict(params, serviceKey=key()), timeout=120)
            txt = r.text
            if "<" not in txt[:50]:
                err = f"XML 아님: {txt[:200]}"
            else:
                root = ET.fromstring(txt.encode("utf-8") if isinstance(txt, str) else txt)
                code = root.findtext(".//resultCode") or root.findtext(".//returnReasonCode")
                msg = root.findtext(".//resultMsg") or root.findtext(".//returnAuthMsg")
                if code in ("00", "0", None) and root.find(".//returnReasonCode") is None:
                    return txt
                if code in ("03",):                    # 데이터 없음
                    return txt
                if code in ("20", "30", "31", "32"):
                    raise SystemExit(f"인증 오류 {code}: {msg} — 승인 직후면 1~2시간 뒤 다시 실행하세요.")
                if code == "22":
                    raise SystemExit(f"일일 호출 한도 초과 ({msg}). 내일 다시 실행하면 이어서 받습니다.")
                if code in ("10", "11", "12", "99"):
                    raise SystemExit(f"요청 오류 {code}: {msg} — 파라미터 확인 ({params})")
                err = f"{code} {msg}"
        except (requests.RequestException, ET.ParseError) as e:
            err = type(e).__name__
        print(f"  재시도 {i + 1}/{retries} ({err})")
        time.sleep(10 * (i + 1))
    raise SystemExit(f"응답 실패: {err}")


def parse(txt):
    root = ET.fromstring(txt.encode("utf-8"))
    rows = [{c.tag: (c.text or "").strip() for c in it} for it in root.iter("item")]
    total = root.findtext(".//totalCount")
    return rows, (int(total) if total and total.isdigit() else None)


def fetch_all(params, rows_per_page=10000):
    """페이지가 있으면 끝까지. 없으면 첫 응답 그대로."""
    out, page = [], 1
    while True:
        txt = call(dict(params, numOfRows=rows_per_page, pageNo=page))
        rows, total = parse(txt)
        out += rows
        if total is None or not rows or len(out) >= total:
            return out, txt
        page += 1
        time.sleep(0.3)


def years(start, end):
    s, e = int(start[:4]), int(end[:4])
    for y in range(s, e + 1):
        yield y, (start if y == s else f"{y}01"), (end if y == e else f"{y}12")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("peek")
    p.add_argument("--cnty", default="US")
    p.add_argument("--ym", default="202501")
    p.add_argument("--hs", default=None)
    g = sub.add_parser("get")
    g.add_argument("--cnty", default="US")
    g.add_argument("--start", default="201801")
    g.add_argument("--end", default="202608")
    g.add_argument("--hs", default=None, help="쉼표로 구분 (예: 87,84). 생략 시 전 품목")
    g.add_argument("--out", default=str(OUT))
    a = ap.parse_args()

    if a.cmd == "peek":
        prm = dict(strtYymm=a.ym, endYymm=a.ym, cntyCd=a.cnty)
        if a.hs:
            prm["hsSgn"] = a.hs
        rows, txt = fetch_all(prm)
        print(f"행 {len(rows)} | 응답 앞부분:\n{txt[:600]}\n")
        if rows:
            d = pd.DataFrame(rows)
            print("컬럼:", list(d.columns))
            if "hsCd" in d:
                print("HS 코드 자릿수 분포:", d["hsCd"].str.len().value_counts().to_dict())
            print(d.head(15).to_string())
        return

    out = Path(a.out)
    (out / "raw").mkdir(parents=True, exist_ok=True)
    hs_list = AUTO_HS4 if a.hs == "auto4" else (a.hs.split(",") if a.hs else [None])
    recs = []
    for c in a.cnty.split(","):
        for hs in hs_list:
            for y, st, en in years(a.start, a.end):
                f = out / "raw" / f"{c}_{y}_{hs or 'all'}.xml"
                if f.exists() and f.stat().st_size > 100 and not (y == int(a.end[:4])):
                    txt = f.read_text(encoding="utf-8")
                    rows, _ = parse(txt)
                else:
                    prm = dict(strtYymm=st, endYymm=en, cntyCd=c)
                    if hs:
                        prm["hsSgn"] = hs
                    rows, txt = fetch_all(prm)
                    f.write_text(txt, encoding="utf-8")
                    time.sleep(0.5)
                print(f"  {c} {hs or '전품목'} {y}: {len(rows):,}행")
                for r in rows:
                    r["cnty"] = c
                recs += rows
    d = pd.DataFrame(recs)
    if d.empty:
        raise SystemExit("받은 행이 없습니다. peek로 응답을 먼저 확인하세요.")
    d = d.rename(columns={"year": "period", "statKor": "item_name", "hsCd": "hs"})
    d = d[d["period"].astype(str).str.match(r"^\d{4}\.?\d{2}$") & (d["hs"].astype(str) != "-")]     # '총계' 행 제거
    d["ym"] = pd.to_datetime(d["period"].astype(str).str.replace(".", "", regex=False), format="%Y%m")
    for c in NUM:
        if c in d:
            d[c] = pd.to_numeric(d[c].astype(str).str.replace(",", ""), errors="coerce")
    d = d.drop_duplicates(["cnty", "hs", "ym"])
    dst = out / "nitem_trade.csv"
    d.to_csv(dst, index=False, encoding="utf-8-sig")
    print(f"\n저장: {dst} ({len(d):,}행, {d['ym'].min():%Y-%m}~{d['ym'].max():%Y-%m}, "
          f"국가 {d['cnty'].nunique()}개, HS {d['hs'].nunique():,}개)")


if __name__ == "__main__":
    main()
