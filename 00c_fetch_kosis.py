"""
00c. KOSIS(국가통계포털) 통계표 수집 → 월별 검증용 공식 통계
    - 지역별 광공업생산지수 (시도 × 산업 × 월): 위성 지표의 시도·업종 합계와 비교
    - 광업제조업동향조사 가동률지수 (산업 × 월): 가동 강도 비교
    - 그 밖에 필요한 통계표 무엇이든 (orgId·tblId만 알면 됨)

사전 준비 (둘 중 하나):
  ① 공공데이터포털 "국가데이터처_KOSIS 통계자료 조회 서비스" 활용신청 → 기존 DATA_GO_KR_KEY 그대로 사용 (해외 접속 가능)
  ② kosis.kr/openapi 인증키 → os.environ["KOSIS_KEY"] (검색 기능은 이 경로에서만)

사용법:
    # 1) 통계표 찾기 (이름으로 검색 → orgId, tblId 확인)
    python 00c_fetch_kosis.py search "광공업생산지수"
    python 00c_fetch_kosis.py search "시도 광공업생산지수"

    # 2) 통계표 구조 보기 (항목·분류 코드 확인, 최근 1개월만 받아 봄)
    python 00c_fetch_kosis.py peek --org 101 --tbl DT_XXXX

    # 3) 받기 (월별, 기간을 1년씩 나눠 요청 → 한 번 요청의 셀 수 제한 회피)
    python 00c_fetch_kosis.py get --org 101 --tbl DT_XXXX --start 201801 --end 202608 \\
        --name regional_ip --out data/raw/kosis

출력: <out>/<name>.csv  (긴 형식: 기간·분류·항목·값, KOSIS 원 컬럼 그대로 + 정리 컬럼)
"""
import argparse
import json
import os
import time
from pathlib import Path

import pandas as pd
import requests

DATA_URL = "https://kosis.kr/openapi/Param/statisticsParameterData.do"
SEARCH_URL = "https://kosis.kr/openapi/statisticsSearch.do"
# 공공데이터포털 경유 (kosis.kr 해외 접속이 막힐 때): 국가데이터처_KOSIS 통계자료 조회 서비스
GW_URL = "https://apis.data.go.kr/1240000/statisticsData/getStatisticsData"
GW_ROWS = 1000


def route():
    """KOSIS_KEY가 있으면 kosis.kr 직접, 없고 DATA_GO_KR_KEY가 있으면 공공데이터포털 경유."""
    if os.environ.get("KOSIS_KEY"):
        return "kosis"
    if os.environ.get("DATA_GO_KR_KEY"):
        return "gw"
    raise SystemExit("환경변수 KOSIS_KEY 또는 DATA_GO_KR_KEY를 넣어주세요.")


def key():
    if route() == "kosis":
        return os.environ["KOSIS_KEY"]
    from urllib.parse import unquote
    return unquote(os.environ["DATA_GO_KR_KEY"])


def gw_items(js):
    """공공데이터포털 응답에서 행 목록과 전체 건수 추출 (KOSIS 원형 리스트도 허용)."""
    if isinstance(js, list):
        return js, len(js)
    auth = js.get("OpenAPI_ServiceResponse", {}).get("cmmMsgHeader")
    if auth:
        raise SystemExit(f"인증 오류 {auth.get('returnReasonCode')}: {auth.get('returnAuthMsg')} "
                         f"— 활용신청 직후면 1~2시간 뒤 다시 실행하세요.")
    r = js.get("response", js)
    hdr = r.get("header", {})
    code = str(hdr.get("resultCode", "00"))
    if code not in ("00", "0", "INFO-000"):
        raise SystemExit(f"공공데이터포털 오류 {code}: {hdr.get('resultMsg')}")
    body = r.get("body", {})
    it = body.get("items", [])
    it = it.get("item", []) if isinstance(it, dict) else it
    it = [it] if isinstance(it, dict) else (it or [])
    return it, int(body.get("totalCount", len(it)) or 0)


def gw_get(params):
    """페이지를 돌며 전부 받기."""
    rows, page = [], 1
    while True:
        p = dict(params, serviceKey=key(), pageNo=page, numOfRows=GW_ROWS, format="json")
        js = call(GW_URL, p)
        if isinstance(js, str):
            raise SystemExit(f"예상과 다른 응답(JSON 아님): {js[:300]}")
        it, total = gw_items(js)
        rows += it
        if not it or len(rows) >= total:
            return rows
        page += 1
        time.sleep(0.3)


def call(url, params, retries=4):
    for i in range(retries):
        try:
            r = requests.get(url, params=params, timeout=120)
            txt = r.text.strip()
            try:
                js = json.loads(txt)
            except ValueError:
                js = None
            if isinstance(js, dict) and js.get("err") and not js.get("response"):
                raise SystemExit(f"KOSIS 오류 {js.get('err')}: {js.get('errMsg')} — 인증키·코드를 확인하세요.")
            return js if js is not None else txt
        except requests.RequestException as e:
            print(f"  재시도 {i + 1}/{retries} ({type(e).__name__})")
            time.sleep(10 * (i + 1))
    raise SystemExit("KOSIS 응답 없음. 잠시 후 다시 실행하세요.")


def search(term):
    if route() == "gw":
        raise SystemExit("검색은 kosis.kr 직접 키(KOSIS_KEY)가 있어야 합니다. 공공데이터포털 경로에서는 "
                         "통계표 ID를 알고 있어야 합니다 (예: 시도/산업별 광공업생산지수 101 / DT_1F02001).")
    js = call(SEARCH_URL, dict(method="getList", apiKey=key(), searchNm=term, format="json",
                               jsonVD="Y", startCount=1, resultCount=30, sort="RANK"))
    if not isinstance(js, list):
        print(js if isinstance(js, str) else json.dumps(js, ensure_ascii=False)[:1000])
        return
    rows = [{k: r.get(k) for k in ("ORG_ID", "TBL_ID", "TBL_NM", "STAT_NM", "STRT_PRD_DE", "END_PRD_DE")} for r in js]
    print(pd.DataFrame(rows).to_string())


def fetch(org, tbl, prd, start, end, itm="ALL", objs=("ALL",)):
    p = dict(orgId=org, tblId=tbl, itmId=itm, prdSe=prd, startPrdDe=start, endPrdDe=end)
    for i, o in enumerate(objs, 1):
        p[f"objL{i}"] = o
    if route() == "gw":
        return pd.DataFrame(gw_get(p))
    p.update(method="getList", apiKey=key(), format="json", jsonVD="Y")
    js = call(DATA_URL, p)
    if not isinstance(js, list):
        raise SystemExit(f"예상과 다른 응답: {str(js)[:500]}")
    return pd.DataFrame(js)


def tidy(df):
    keep = [c for c in df.columns if c in ("PRD_DE", "ITM_ID", "ITM_NM", "UNIT_NM", "DT")
            or c.startswith("C") and (c[1:].isdigit() or c.endswith("_NM"))]
    out = df[keep].copy()
    out["value"] = pd.to_numeric(out["DT"].astype(str).str.replace(",", ""), errors="coerce")
    if "PRD_DE" in out:
        p = out["PRD_DE"].astype(str)
        out["ym"] = pd.to_datetime(p.str[:6], format="%Y%m", errors="coerce") if p.str.len().max() >= 6 else None
    return out


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("search"); s.add_argument("term")
    for name in ("peek", "get"):
        g = sub.add_parser(name)
        g.add_argument("--org", required=True)
        g.add_argument("--tbl", required=True)
        g.add_argument("--prd", default="M", help="M 월, Q 분기, Y 연")
        g.add_argument("--itm", default="ALL")
        g.add_argument("--obj", nargs="*", default=["ALL", "ALL"],
                       help="분류 단계별 코드 (objL1 objL2 ...). 기본: 1·2단계 전체")
        if name == "get":
            g.add_argument("--start", required=True)
            g.add_argument("--end", required=True)
            g.add_argument("--name", required=True)
            g.add_argument("--out", default="data/raw/kosis")
    a = ap.parse_args()

    if a.cmd == "search":
        return search(a.term)

    if a.cmd == "peek":
        p = dict(orgId=a.org, tblId=a.tbl, itmId=a.itm, prdSe=a.prd, newEstPrdCnt=1)
        for i, o in enumerate(a.obj, 1):
            p[f"objL{i}"] = o
        if route() == "gw":
            js = gw_get(p)
        else:
            p.update(method="getList", apiKey=key(), format="json", jsonVD="Y")
            js = call(DATA_URL, p)
        df = pd.DataFrame(js) if isinstance(js, list) else None
        if df is not None and df.empty:
            raise SystemExit("결과 0행 — 통계표 ID·분류 코드(objL)·항목(itmId)을 확인하세요.")
        if df is None:
            raise SystemExit(f"예상과 다른 응답: {str(js)[:500]}")
        print(f"행 {len(df)}, 기간 {df['PRD_DE'].unique()[:3]}")
        print("컬럼:", list(df.columns))
        for c in [c for c in df.columns if c.endswith("_NM")]:
            vals = df[c].unique()
            print(f"  {c} ({len(vals)}종): {list(vals[:15])}")
        return

    # get: 1년씩 나눠 받기
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    s_y, e_y = int(a.start[:4]), int(a.end[:4])
    parts = []
    for y in range(s_y, e_y + 1):
        st = a.start if y == s_y else f"{y}01"
        en = a.end if y == e_y else f"{y}12"
        if a.prd == "Q":
            st, en = (a.start if y == s_y else f"{y}01"), (a.end if y == e_y else f"{y}04")
        if a.prd == "Y":
            st = en = str(y)
        df = fetch(a.org, a.tbl, a.prd, st, en, a.itm, a.obj)
        print(f"  {st}~{en}: {len(df):,}행")
        parts.append(df)
        time.sleep(0.5)
    raw = pd.concat(parts, ignore_index=True).drop_duplicates()
    td = tidy(raw)
    f = out / f"{a.name}.csv"
    td.to_csv(f, index=False, encoding="utf-8-sig")
    print(f"\n저장: {f} ({len(td):,}행, 기간 {td['PRD_DE'].min()}~{td['PRD_DE'].max()})")
    print(td.head().to_string())


if __name__ == "__main__":
    main()
