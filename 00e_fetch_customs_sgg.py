"""
00e. 관세청 시군구별 품목별 수출입실적 (공공데이터포털 15134343) → 지역 × HS6 × 월 수출
API: https://apis.data.go.kr/1220000/sigunguperprlstperacrs/getSigunguPerPrlstPerAcrs
     serviceKey, strtYymm, endYymm, HsSgn (HS 6단위), sidoCd (시도코드)
특징: 수출 지역 = 제조장소 우편번호 기준 → 공장 위치와 직접 연결 가능. 상대국 구분은 없음.
주의: 2026-07 행정구역 개편으로 시군구 코드가 바뀜 (관세청조회코드_v1.3.xlsx 참고)

사용법:
    !python 00e_fetch_customs_sgg.py peek --sido 31 --hs 870323 --ym 202501
    !python 00e_fetch_customs_sgg.py peek --sido 31 --ym 202501            # HsSgn 생략 가능한지 확인
    !python 00e_fetch_customs_sgg.py get --hs auto --start 201801 --end 202608
출력: data/raw/customs_sgg/raw/<시도>_<HS>_<연도>.xml , data/raw/customs_sgg/sgg_hs_trade.csv
"""
import argparse
import os
import time
import xml.etree.ElementTree as ET
from pathlib import Path
from urllib.parse import unquote

import pandas as pd
import requests

URL = "https://apis.data.go.kr/1220000/sigunguperprlstperacrs/getSigunguPerPrlstPerAcrs"
OUT = Path("data/raw/customs_sgg")
# 관세청 시도코드 (2026-10 확인): 강원 51·전북 52 (옛 42·45는 과거 자료도 오류),
# 전남광주통합특별시 12 (2026-07 이후 자료만). 광주 29·전남 46은 2026-06까지.
SIDO = {"11": "서울", "26": "부산", "27": "대구", "28": "인천", "29": "광주", "30": "대전", "31": "울산",
        "36": "세종", "41": "경기", "51": "강원", "43": "충북", "44": "충남", "52": "전북", "46": "전남",
        "47": "경북", "48": "경남", "50": "제주", "12": "전남광주"}
FIRST_YEAR = {"12": 2026}          # 이 코드는 해당 연도부터만 요청
GWANGJU_GU = {"광산구", "북구", "서구", "남구", "동구"}
SHORT = [("서울", "서울"), ("부산", "부산"), ("대구", "대구"), ("인천", "인천"), ("광주광역", "광주"),
         ("대전", "대전"), ("울산", "울산"), ("세종", "세종"), ("경기", "경기"), ("강원", "강원"),
         ("충청북", "충북"), ("충북", "충북"), ("충청남", "충남"), ("충남", "충남"), ("전라북", "전북"),
         ("전북", "전북"), ("전라남", "전남"), ("경상북", "경북"), ("경북", "경북"), ("경상남", "경남"),
         ("경남", "경남"), ("제주", "제주")]
# 자동차 공급망 HS6 (완성차·부품·타이어·엔진 등). 'auto'로 지정하면 이 목록 사용
AUTO_HS6 = [
    "870321", "870322", "870323", "870324", "870331", "870332", "870333", "870340", "870350", "870360",
    "870370", "870380", "870390", "870421", "870422", "870423", "870431", "870490", "870600", "870710",
    "870810", "870821", "870829", "870830", "870840", "870850", "870870", "870880", "870891", "870892",
    "870893", "870894", "870895", "870899", "840731", "840732", "840733", "840734", "840820", "840991",
    "840999", "401110", "401120", "850760", "851110", "851140", "851180", "851220", "851230", "851240",
    "853690", "854430", "940120", "830230", "848340", "848310", "732020", "700711", "700721", "701091",
]


def region(name):
    """'전남광주통합특별시 광산구' → ('광주','광산구'), '전라남도 여수시' → ('전남','여수시')."""
    t = str(name).split()
    if not t:
        return None, None
    a, rest = t[0], " ".join(t[1:])
    if a.startswith("전남광주"):
        return ("광주" if t[1:2] and t[1] in GWANGJU_GU else "전남"), rest
    for k, v in SHORT:
        if a.startswith(k):
            return v, rest
    return None, rest


def tidy(d):
    for c in ("expCnt", "expUsdAmt", "impCnt", "impUsdAmt", "cmtrBlncAmt"):
        if c in d:
            d[c] = pd.to_numeric(d[c].astype(str).str.replace(",", ""), errors="coerce")
    d = d[d["priodTitle"].astype(str).str.match(r"^\d{4}\.\d{2}$")].copy()
    d["ym"] = pd.to_datetime(d["priodTitle"].str.replace(".", "", regex=False), format="%Y%m")
    rs = d["sggNm"].map(region)
    d["sido"] = [r[0] for r in rs]
    d["sgg"] = [r[1] for r in rs]
    # 개편 전후 코드가 겹쳐도 (시도, 시군구, HS, 월) 하나만
    d = d.drop_duplicates(["sido", "sgg", "hsSgn", "ym"])
    return d


def key():
    k = os.environ.get("DATA_GO_KR_KEY")
    if not k or len(k) < 30:
        raise SystemExit("환경변수 DATA_GO_KR_KEY에 실제 인증키를 넣어주세요.")
    return unquote(k)


def call(params, retries=5):
    err = None
    for i in range(retries):
        try:
            txt = requests.get(URL, params=dict(params, serviceKey=key()), timeout=120).text
            if "<" not in txt[:50]:
                err = f"XML 아님: {txt[:200]}"
            else:
                root = ET.fromstring(txt.encode("utf-8"))
                code = root.findtext(".//resultCode") or root.findtext(".//returnReasonCode")
                msg = root.findtext(".//resultMsg") or root.findtext(".//returnAuthMsg")
                if code in ("00", "0", None, "03") and root.find(".//returnReasonCode") is None:
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


def fetch_all(params, n=10000):
    out, page = [], 1
    while True:
        txt = call(dict(params, numOfRows=n, pageNo=page))
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
    p.add_argument("--sido", default="31"); p.add_argument("--hs", default=None); p.add_argument("--ym", default="202501")
    g = sub.add_parser("get")
    g.add_argument("--sido", default=",".join(SIDO))
    g.add_argument("--hs", default="auto", help="'auto' 또는 쉼표로 구분한 HS6")
    g.add_argument("--start", default="201801"); g.add_argument("--end", default="202608")
    g.add_argument("--out", default=str(OUT))
    a = ap.parse_args()

    if a.cmd == "peek":
        prm = dict(strtYymm=a.ym, endYymm=a.ym, sidoCd=a.sido)
        if a.hs:
            prm["HsSgn"] = a.hs
        rows, txt = fetch_all(prm)
        print(f"행 {len(rows)} | 응답 앞부분:\n{txt[:800]}\n")
        if rows:
            d = pd.DataFrame(rows)
            print("컬럼:", list(d.columns))
            print(d.head(20).to_string())
        return

    out = Path(a.out); (out / "raw").mkdir(parents=True, exist_ok=True)
    hs_list = AUTO_HS6 if a.hs == "auto" else a.hs.split(",")
    sidos = a.sido.split(",")
    n_call = len(hs_list) * sum(1 for s in sidos for y, _, _ in years(a.start, a.end)
                                if y >= FIRST_YEAR.get(s, 0))
    print(f"요청 {n_call:,}건 예정 (일일 한도 10,000건; 받은 것은 캐시돼 다음 실행 때 건너뜀)")
    recs = []
    for s in sidos:
        for hs in hs_list:
            for y, st, en in years(a.start, a.end):
                if y < FIRST_YEAR.get(s, 0):
                    continue
                f = out / "raw" / f"{s}_{hs}_{y}.xml"
                if f.exists() and f.stat().st_size > 100 and y != int(a.end[:4]):
                    rows, _ = parse(f.read_text(encoding="utf-8"))
                else:
                    rows, txt = fetch_all(dict(strtYymm=st, endYymm=en, HsSgn=hs, sidoCd=s))
                    f.write_text(txt, encoding="utf-8")
                    time.sleep(0.2)
                for r in rows:
                    r.update(q_sido=s, q_hs=hs)
                recs += rows
        print(f"  {s} {SIDO.get(s, '')}: 누적 {len(recs):,}행")
    d = pd.DataFrame(recs)
    if d.empty:
        raise SystemExit("받은 행이 없습니다. peek로 응답과 시도코드를 확인하세요.")
    d = tidy(d)
    dst = out / "sgg_hs_trade.csv"
    d.to_csv(dst, index=False, encoding="utf-8-sig")
    print(f"\n저장: {dst} ({len(d):,}행, {d['ym'].min():%Y-%m}~{d['ym'].max():%Y-%m}, "
          f"시군구 {d[['sido','sgg']].drop_duplicates().shape[0]}개, HS6 {d['hsSgn'].nunique()}개)")
    print("시도별 수출 합계(천 달러, 마지막 12개월):")
    last = d[d["ym"] > d["ym"].max() - pd.DateOffset(months=12)]
    print(last.groupby("sido")["expUsdAmt"].sum().sort_values(ascending=False).round(0).to_string())


if __name__ == "__main__":
    main()
