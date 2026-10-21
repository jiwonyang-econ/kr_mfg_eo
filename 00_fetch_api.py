"""
00. 공공데이터포털 API 전체 수집 (Colab 또는 내 컴퓨터에서 실행)

- 페이지마다 원본 XML을 저장 → 중간에 끊겨도 다시 실행하면 받은 페이지는 건너뜀
- 타임아웃·서버 오류는 대기 후 재시도
- 끝나면 모든 페이지를 합쳐 CSV 하나로 저장 (공장관리번호 기준 중복 제거)

Colab 사용법:
    from google.colab import drive; drive.mount('/content/drive')   # 받은 페이지를 드라이브에 보관
    import os; os.environ["DATA_GO_KR_KEY"] = "일반인증키"
    !python 00_fetch_api.py lndpcl --out /content/drive/MyDrive/kr_mfg_eo/api

명령:
    python 00_fetch_api.py lndpcl                 # 공장등록필지정보 (cmpnyNm='%'로 전체)
    python 00_fetch_api.py lndpcl --test          # 3페이지만 받아 보기
    python 00_fetch_api.py lndpcl --merge-only    # 이미 받은 페이지로 CSV만 다시 만들기
    python 00_fetch_api.py prdctn --rows 100      # 서버가 큰 페이지를 못 버틸 때 (오류 05)
"""
import argparse
import os
import time
import xml.etree.ElementTree as ET
from pathlib import Path
from urllib.parse import unquote

import pandas as pd
import requests

SERVICES = {
    # 이름: (요청주소, 고정 요청변수, 중복제거 키)
    "lndpcl": ("https://apis.data.go.kr/B550624/fctryRegistLndpclInfo/getFctryLndpclService",
               {"cmpnyNm": "%"}, "fctryManageNo"),
    # 생산정보: 공장 하나에 생산품·업종이 여러 개일 수 있으므로 공장관리번호로 중복 제거하지 않음
    #   (완전히 같은 행만 제거). 01 단계에서 공장 단위로 묶음.
    "prdctn": ("https://apis.data.go.kr/B550624/fctryRegistInfo/getFctryPrdctnService_v2",
               {"cmpnyNm": "%"}, None),
    # 산업동향 API는 명세 확인 후 여기에 추가
}
TRY_ROWS = [1000, 500, 100]   # 한 페이지 건수: 큰 것부터 시도
TIMEOUT = 120
MAX_RETRY = 6


def get_key():
    k = os.environ.get("DATA_GO_KR_KEY")
    if not k:
        raise SystemExit("환경변수 DATA_GO_KR_KEY에 일반인증키를 넣어주세요.")
    return unquote(k)   # Encoding 형태(%2B…)로 넣어도 자동 변환


def call(url, params):
    """요청 1회 (재시도 포함). 반환: (XML 텍스트, resultCode, totalCount, item 수)"""
    for attempt in range(1, MAX_RETRY + 1):
        try:
            r = requests.get(url, params=params, timeout=TIMEOUT)
            root = ET.fromstring(r.text)
            # 인증·주소 오류는 다른 형식(OpenAPI_ServiceResponse/returnReasonCode)으로 옴
            code = root.findtext(".//resultCode") or root.findtext(".//returnReasonCode")
            if code == "00":
                total = int(root.findtext(".//totalCount") or 0)
                return r.text, code, total, len(root.findall(".//item"))
            msg = root.findtext(".//resultMsg") or root.findtext(".//returnAuthMsg")
            if code in ("22", "LIMITED_NUMBER_OF_SERVICE_REQUESTS_EXCEEDS_ERROR"):
                raise SystemExit(f"일일 호출 한도 초과({msg}). 내일 다시 실행하면 이어서 받습니다.")
            if code in ("10", "11", "12", "20", "30", "31", "32"):
                raise SystemExit(f"요청 오류 resultCode={code} {msg} — 키·주소·변수를 확인하세요.")
            err = f"resultCode={code} {msg}"
        except (requests.RequestException, ET.ParseError) as e:
            err = f"{type(e).__name__}"
        wait = min(10 * 2 ** (attempt - 1), 300)
        print(f"  재시도 {attempt}/{MAX_RETRY} ({err}) — {wait}초 대기")
        time.sleep(wait)
    raise SystemExit("재시도 한도 초과. 잠시 후 다시 실행하면 받은 페이지부터 이어서 진행합니다.")


def choose_rows(url, base, key, out):
    """서버가 실제로 허용하는 페이지 크기를 확인해 고정 (페이지 번호 체계가 바뀌지 않도록)."""
    f = out / "_rows.txt"
    if f.exists():
        return int(f.read_text())
    for rows in TRY_ROWS:
        _, _, total, n = call(url, {**base, "serviceKey": key, "pageNo": 1, "numOfRows": rows})
        if n == rows or n == total:
            f.write_text(str(rows))
            print(f"페이지 크기 {rows}건으로 고정 (전체 {total:,}건)")
            return rows
        print(f"numOfRows={rows} 요청 → {n}건만 반환, 더 작은 크기로 시도")
    raise SystemExit("적절한 페이지 크기를 찾지 못했습니다.")


def fetch(name, out, test=False, rows=None):
    url, base, _ = SERVICES[name]
    key = get_key()
    out.mkdir(parents=True, exist_ok=True)
    if rows:
        # 페이지 크기 직접 지정: 크기가 바뀌면 페이지 번호가 가리키는 범위가 달라지므로 기존 페이지는 보관 폴더로 옮김
        saved = (out / "_rows.txt").read_text().strip() if (out / "_rows.txt").exists() else None
        old = list(out.glob("page_*.xml"))
        if saved and int(saved) != rows and old:
            bak = out / f"_old_rows{saved}"
            bak.mkdir(exist_ok=True)
            for f in old:
                f.rename(bak / f.name)
            print(f"페이지 크기 {saved}→{rows} 변경: 기존 페이지 {len(old)}개를 {bak.name}/ 로 옮김")
        (out / "_rows.txt").write_text(str(rows))
    rows = choose_rows(url, base, key, out)
    _, _, total, _ = call(url, {**base, "serviceKey": key, "pageNo": 1, "numOfRows": rows})
    n_pages = -(-total // rows)
    last = min(3, n_pages) if test else n_pages
    print(f"[{name}] 전체 {total:,}건 / {n_pages:,}페이지 (페이지당 {rows})")
    for p in range(1, last + 1):
        f = out / f"page_{p:05d}.xml"
        if f.exists():
            continue
        text, _, _, n = call(url, {**base, "serviceKey": key, "pageNo": p, "numOfRows": rows})
        f.write_text(text, encoding="utf-8")
        if p % 20 == 0 or p == last:
            print(f"  {p:,}/{last:,} 페이지 저장")
        time.sleep(1.5)   # 공단 서버 부담을 줄이려 페이지 사이 간격을 둠
    (out / "_meta.txt").write_text(f"service={name}\ntotal={total}\nrows={rows}\n"
                                   f"fetched_at={pd.Timestamp.now():%Y-%m-%d %H:%M}\n")


def flatten_item(it):
    """<item>의 하위 항목을 한 행으로. 하위에 또 목록이 있으면(예: 생산품 여러 개)
    '상위.하위' 컬럼에 값을 ' | '로 이어 붙임 → 정보 손실 없이 CSV 한 행에 담음."""
    rec = {}
    for c in it:
        if len(c) == 0:
            rec[c.tag] = (c.text or "").strip()
            continue
        for g in c.iter():
            if g is c or len(g):
                continue
            k = f"{c.tag}.{g.tag}"
            v = (g.text or "").strip()
            rec[k] = f"{rec[k]} | {v}" if k in rec else v
    return rec


def merge(name, out):
    _, _, dedup = SERVICES[name]
    recs = []
    for f in sorted(out.glob("page_*.xml")):
        for it in ET.parse(f).getroot().iter("item"):
            recs.append(flatten_item(it))
    df = pd.DataFrame(recs)
    n0 = len(df)
    if dedup and dedup in df:
        df = df.drop_duplicates(subset=dedup)
    else:
        df = df.drop_duplicates()
    meta = dict(l.split("=", 1) for l in (out / "_meta.txt").read_text().splitlines() if "=" in l) \
        if (out / "_meta.txt").exists() else {}
    stamp = meta.get("fetched_at", "")[:10].replace("-", "")
    csv = out.parent / f"{name}_{stamp or 'snapshot'}.csv"
    df.to_csv(csv, index=False, encoding="utf-8-sig")
    print(f"\n병합: {n0:,}행 → 중복 제거 {len(df):,}행 (서버 totalCount {meta.get('total', '?')})")
    print(f"컬럼: {list(df.columns)}")
    print(f"저장: {csv}")
    return csv


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("service", choices=list(SERVICES))
    ap.add_argument("--out", default="data/raw/api")
    ap.add_argument("--test", action="store_true")
    ap.add_argument("--merge-only", action="store_true")
    ap.add_argument("--rows", type=int, default=None, help="페이지 크기 직접 지정 (서버 오류 05가 계속되면 100 또는 200)")
    a = ap.parse_args()
    out = Path(a.out) / a.service
    if not a.merge_only:
        fetch(a.service, out, a.test, a.rows)
    merge(a.service, out)


if __name__ == "__main__":
    main()
