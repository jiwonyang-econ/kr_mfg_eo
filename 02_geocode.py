"""
02. 공장 주소 지오코딩 (카카오 로컬 API 기본, 브이월드 선택)
입력: data/interim/factories_clean.csv
출력: data/interim/factories_geocoded.csv

사용법:
    export KAKAO_KEY=REST_API_키        # developers.kakao.com 앱 → REST API 키, 앱 설정에서 '카카오맵' 사용 ON
    python 02_geocode.py --large-only    # 대형공장만 (파일럿)
    python 02_geocode.py                 # 전체
    python 02_geocode.py --workers 8     # 동시 호출 수 (기본 8)
    python 02_geocode.py --mock          # 키 없이 테스트 (가짜 좌표)
    (브이월드: --provider vworld, export VWORLD_KEY=키 — 해외 접속이 막힐 수 있음)

단계
  1) 주소검색: 정리된 주소(geo_addr)로 좌표 찾기. 주소 단위 캐시(geocode_cache.csv)에 누적
  2) 키워드 보완 (대형공장만, 카카오만): 주소가 없거나 1)에서 못 찾은 대형공장을
     '회사명 + 시군구(또는 산업단지명)'로 장소 검색. 결과는 geo_status='keyword'로 표시하고,
     주소가 있는 경우 검색된 장소가 같은 시군구에 있을 때만 채택 (본사 사무실 등 오매칭 방지)

- 캐시가 있어서 중간에 끊겨도 다시 실행하면 이어서 진행
- 네트워크 오류(error:*)는 캐시에 남기지 않아 다음 실행에서 다시 시도
"""
import argparse
import hashlib
import os
import re
from concurrent.futures import ThreadPoolExecutor

import pandas as pd
import requests
from config import INT, VWORLD_KEY_ENV, GEOCODE_CACHE
from utils import norm_complex_name

VWORLD_URL = "https://api.vworld.kr/req/address"
KAKAO_ADDR_URL = "https://dapi.kakao.com/v2/local/search/address.json"
KAKAO_KW_URL = "https://dapi.kakao.com/v2/local/search/keyword.json"
KEYWORD_CACHE = INT / "keyword_cache.csv"
PLANT_WORDS = r"공장|제철소|조선소|사업장|사업소|캠퍼스|공단|산업단지|생산|플랜트|정유|제련|단지"
SAVE_EVERY = 500


class QuotaExceeded(Exception):
    pass


def _kakao_get(url, params, key):
    r = requests.get(url, params=params, headers={"Authorization": f"KakaoAK {key}"}, timeout=15)
    # 카카오는 일일 한도를 넘기면 429 또는 400/403 + 'limit' 메시지로 응답함
    if r.status_code == 429 or (r.status_code in (400, 403) and "limit" in r.text.lower()):
        raise QuotaExceeded(r.text[:200])
    r.raise_for_status()
    return r.json().get("documents", [])


def kakao_addr(addr, key):
    docs = _kakao_get(KAKAO_ADDR_URL, {"query": addr, "size": 1}, key)
    return (float(docs[0]["x"]), float(docs[0]["y"])) if docs else None


def vworld_addr(addr, key, addr_type="road"):
    params = dict(service="address", request="getcoord", version="2.0", crs="epsg:4326",
                  address=addr, refine="true", simple="false", format="json", type=addr_type, key=key)
    r = requests.get(VWORLD_URL, params=params, timeout=15)
    r.raise_for_status()
    js = r.json().get("response", {})
    if js.get("status") != "OK":
        return None
    pt = js["result"]["point"]
    return float(pt["x"]), float(pt["y"])


def mock_addr(addr, key=None):
    """테스트용: 주소 해시로 한반도 남부 범위 안의 결정적 좌표. '0번지'가 든 주소는 실패로 처리."""
    if "0번지" in addr:
        return None
    h = int(hashlib.md5(addr.encode()).hexdigest(), 16)
    return 126.5 + (h % 10_000) / 10_000 * 2.8, 34.8 + (h // 10_000 % 10_000) / 10_000 * 3.0


def run_parallel(items, fn, workers, label, on_batch):
    """items를 SAVE_EVERY개씩 나눠 병렬 처리하고, 묶음마다 on_batch(결과)로 저장."""
    for s in range(0, len(items), SAVE_EVERY):
        chunk = items[s:s + SAVE_EVERY]
        with ThreadPoolExecutor(max_workers=workers) as ex:
            res = list(ex.map(fn, chunk))
        on_batch(res)
        print(f"  [{label}] {min(s + SAVE_EVERY, len(items)):,}/{len(items):,} 저장")


def load_csv(path, cols):
    return pd.read_csv(path, dtype=str) if path.exists() else pd.DataFrame(columns=cols)


# ------------------------------------------------------------------ 1) 주소검색
def geocode_addresses(todo, addr_fn, key, workers):
    cache = load_csv(GEOCODE_CACHE, ["geo_addr", "lon", "lat", "geo_status"])
    cache = cache[~cache["geo_status"].fillna("").str.startswith("error")]   # 오류는 재시도
    queue = todo.loc[~todo["geo_addr"].isin(set(cache["geo_addr"])), ["geo_addr", "geo_addr_type"]] \
                .drop_duplicates("geo_addr")
    print(f"[주소검색] 대상 주소 {todo['geo_addr'].nunique():,}개 중 캐시 {todo['geo_addr'].nunique() - len(queue):,}, "
          f"신규 {len(queue):,}")

    def one(rec):
        addr, typ = rec
        try:
            xy = addr_fn(addr, key)
            return dict(geo_addr=addr, lon=xy and xy[0], lat=xy and xy[1], geo_status=typ if xy else "fail")
        except QuotaExceeded:
            return dict(geo_addr=addr, lon=None, lat=None, geo_status="error:quota")
        except requests.HTTPError as e:
            code = e.response.status_code if e.response is not None else "?"
            return dict(geo_addr=addr, lon=None, lat=None, geo_status=f"error:HTTP{code}",
                        _msg=(e.response.text[:200] if e.response is not None else ""))
        except requests.RequestException as e:
            return dict(geo_addr=addr, lon=None, lat=None, geo_status=f"error:{type(e).__name__}")

    state = {"cache": cache}

    def save(res):
        msgs = [r.pop("_msg", "") for r in res]
        state["cache"] = pd.concat([state["cache"], pd.DataFrame(res)], ignore_index=True)
        state["cache"].to_csv(GEOCODE_CACHE, index=False, encoding="utf-8-sig")
        if any(r["geo_status"] == "error:quota" for r in res):
            raise SystemExit("카카오 일일 호출 한도 초과 (한국시간 자정에 초기화). 그 뒤 다시 실행하면 이어서 진행합니다.")
        n_err = sum(str(r["geo_status"]).startswith("error") for r in res)
        if n_err > len(res) * 0.5:   # 절반 넘게 오류면 계속 호출해도 의미 없음 → 멈추고 원인 표시
            sample = next((m for m in msgs if m), "")
            raise SystemExit(f"이번 묶음 {len(res)}건 중 {n_err}건 오류 → 중단합니다. "
                             f"오류 예: {res[0]['geo_status']} {sample}\n"
                             f"한도 초과면 한국시간 자정 이후, 일시 장애면 잠시 후 다시 실행하세요.")

    items = list(queue.itertuples(index=False, name=None))
    if items:
        run_parallel(items, one, workers, "주소", save)
    return state["cache"]


# ------------------------------------------------------------------ 2) 키워드 보완
def firm_core(name):
    """'(주)포스코 광양제철소' → '포스코 광양제철소', '삼성중공업(주)거제조선소' → '삼성중공업 거제조선소'."""
    n = re.sub(r"\(주\)|㈜|주식회사|\(유\)|유한회사|\(재\)|\(사\)", " ", str(name))
    n = re.sub(r"_?\(\d.*?\)|_", " ", n)          # '_(1~4공장)' 같은 꼬리
    return re.sub(r"\s+", " ", n).strip()


def sigungu(addr):
    """주소에서 '시도 시군구' 앞부분 (검색어·검증용)."""
    if not isinstance(addr, str):
        return ""
    t = addr.split()
    return " ".join(t[:2]) if len(t) >= 2 else ""


def keyword_queries(row):
    core = firm_core(row["firm_name"])
    if not core or core in ("-",):
        return []
    q = []
    region = sigungu(row.get("addr_road") if isinstance(row.get("addr_road"), str) else row.get("geo_addr"))
    if region:
        q.append(f"{region.split()[-1]} {core}")          # '포항시 포스코'
    if isinstance(row.get("complex_name"), str) and row["complex_name"]:
        q.append(f"{core} {norm_complex_name(row['complex_name'])[:2]}")   # '포스코 광양'
        q.append(f"{core} {row['complex_name']}")           # '포스코 광양국가산업단지'
    q.append(core)
    return list(dict.fromkeys(q))


def keyword_fill(fac, geo, key, workers, mock=False):
    need = fac[fac["large_plant"] & geo["lon"].isna()].copy()
    if need.empty:
        return geo
    kc = load_csv(KEYWORD_CACHE, ["plant_id", "kw_query", "kw_place", "kw_addr", "kw_lon", "kw_lat"])
    need = need[~need["plant_id"].isin(set(kc["plant_id"]))]
    print(f"[키워드 보완] 좌표 없는 대형공장 {int((fac['large_plant'] & geo['lon'].isna()).sum()):,}건 "
          f"중 신규 검색 {len(need):,}건")

    def one(row):
        region = sigungu(row.get("addr_road")) if isinstance(row.get("addr_road"), str) else ""
        for q in keyword_queries(row):
            try:
                if mock:
                    docs = [{"x": "127.7", "y": "34.9", "place_name": q, "address_name": region + " 어딘가"}]
                else:
                    docs = _kakao_get(KAKAO_KW_URL, {"query": q, "size": 5}, key)
            except QuotaExceeded:
                return None
            except requests.RequestException:
                continue
            tok = firm_core(row["firm_name"]).split()[0] if firm_core(row["firm_name"]) else ""
            cx = row.get("complex_name")
            ctok = norm_complex_name(cx)[:2] if isinstance(cx, str) and cx else ""
            ok = []
            for d in docs:
                place = d.get("place_name", "").replace(" ", "")
                # (1) 장소 이름에 회사명 첫 단어가 들어 있어야 함 (예: '포스코' ⊂ '포스코 광양제철소')
                if len(tok) >= 2 and tok not in place:
                    continue
                # (2) 주소가 있는 공장은 같은 시군구 결과만 (본사 사무실 오매칭 방지)
                if region and region.split()[-1] not in d.get("address_name", ""):
                    continue
                # (2') 주소가 없으면 산업단지명의 지역명(예: '광양국가산업단지' → '광양')이 결과 주소에 있어야 함
                if not region and ctok and ctok not in d.get("address_name", "") + d.get("place_name", ""):
                    continue
                ok.append(d)
            # (3) 공장다운 결과 우선: 이름에 공장·제철소 등, 또는 카카오 업종이 제조업
            plantlike = [d for d in ok if re.search(PLANT_WORDS, d.get("place_name", ""))
                         or "제조" in d.get("category_name", "")]
            # 주소가 없는 공장은 '공장다운' 결과만 채택 (지역 검증을 할 수 없으므로)
            pick = plantlike[0] if plantlike else (ok[0] if ok and region else None)
            if pick is not None:
                d = pick
                return dict(plant_id=row["plant_id"], kw_query=q, kw_place=d.get("place_name"),
                            kw_addr=d.get("address_name"), kw_lon=d["x"], kw_lat=d["y"])
        return dict(plant_id=row["plant_id"], kw_query="", kw_place="", kw_addr="", kw_lon=None, kw_lat=None)

    state = {"kc": kc}

    def save(res):
        if any(r is None for r in res):
            res = [r for r in res if r is not None]
            state["kc"] = pd.concat([state["kc"], pd.DataFrame(res)], ignore_index=True)
            state["kc"].to_csv(KEYWORD_CACHE, index=False, encoding="utf-8-sig")
            raise SystemExit("카카오 일일 호출 한도 초과. 내일 다시 실행하면 이어서 진행합니다.")
        state["kc"] = pd.concat([state["kc"], pd.DataFrame(res)], ignore_index=True)
        state["kc"].to_csv(KEYWORD_CACHE, index=False, encoding="utf-8-sig")

    rows = [r for _, r in need.iterrows()]
    if rows:
        run_parallel(rows, one, workers, "키워드", save)

    kc = state["kc"].drop_duplicates("plant_id", keep="last")
    kc = kc[kc["kw_lon"].notna() & (kc["kw_lon"].astype(str) != "")]
    geo = geo.merge(kc[["plant_id", "kw_lon", "kw_lat", "kw_place", "kw_addr"]], on="plant_id", how="left")
    fill = geo["lon"].isna() & geo["kw_lon"].notna()
    geo.loc[fill, "lon"] = pd.to_numeric(geo.loc[fill, "kw_lon"])
    geo.loc[fill, "lat"] = pd.to_numeric(geo.loc[fill, "kw_lat"])
    geo.loc[fill, "geo_status"] = "keyword"
    return geo.drop(columns=["kw_lon", "kw_lat"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--large-only", action="store_true")
    ap.add_argument("--mock", action="store_true")
    ap.add_argument("--provider", choices=["kakao", "vworld"], default="kakao")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--no-keyword", action="store_true", help="키워드 보완 단계 생략")
    args = ap.parse_args()

    fac = pd.read_csv(INT / "factories_clean.csv", dtype={"geo_addr": str, "factory_no": str})
    scope = fac[fac["large_plant"]] if args.large_only else fac
    todo = scope.dropna(subset=["geo_addr"])

    if args.mock:
        addr_fn, key, workers = mock_addr, None, 1
    else:
        env = "KAKAO_KEY" if args.provider == "kakao" else VWORLD_KEY_ENV
        key = os.environ.get(env)
        if not key:
            raise SystemExit(f"환경변수 {env}에 API 키를 넣어주세요 (또는 --mock).")
        addr_fn = kakao_addr if args.provider == "kakao" else vworld_addr
        workers = args.workers

    cache = geocode_addresses(todo, addr_fn, key, workers)
    cache = cache[~cache["geo_status"].fillna("").str.startswith("error")].drop_duplicates("geo_addr", keep="last")
    geo = fac.merge(cache, on="geo_addr", how="left")
    geo["lon"] = pd.to_numeric(geo["lon"], errors="coerce")
    geo["lat"] = pd.to_numeric(geo["lat"], errors="coerce")
    geo.loc[geo["geo_addr"].isna(), "geo_status"] = "no_address"

    if not args.no_keyword and (args.provider == "kakao" or args.mock):
        geo = keyword_fill(fac, geo, key, workers, mock=args.mock)
        geo["lon"] = pd.to_numeric(geo["lon"], errors="coerce")
        geo["lat"] = pd.to_numeric(geo["lat"], errors="coerce")

    geo.to_csv(INT / "factories_geocoded.csv", index=False, encoding="utf-8-sig")

    sub = geo[geo["large_plant"]] if args.large_only else geo
    ok = sub["lon"].notna()
    print(f"\n성공률: {ok.mean():.1%} ({ok.sum():,}/{len(sub):,})")
    print(sub["geo_status"].fillna("미처리").value_counts().to_string())
    bad = sub[ok & (~sub["lon"].between(124, 132) | ~sub["lat"].between(33, 39))]
    if len(bad):
        print(f"경고: 한국 범위 밖 좌표 {len(bad)}건 → 수동 확인 필요")
    if "keyword" in set(sub["geo_status"]):
        print("\n키워드로 찾은 공장 (검색어 → 찾은 장소) — 오매칭이 없는지 눈으로 확인하세요:")
        kw = sub[sub["geo_status"] == "keyword"]
        print(kw[["firm_name", "kw_place", "kw_addr"]].head(15).to_string())


if __name__ == "__main__":
    main()
