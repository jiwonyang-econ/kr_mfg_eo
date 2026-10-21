"""공통 유틸: 원자료 읽기(인코딩 자동), 컬럼 별칭 매핑."""
import re
import pandas as pd
from config import KSIC2_GROUP


def read_table(path):
    """CSV(cp949/utf-8 자동) 또는 XLSX를 읽는다. 공공데이터포털 CSV는 대개 cp949."""
    path = str(path)
    if path.lower().endswith((".xlsx", ".xls")):
        return pd.read_excel(path, dtype=str)
    for enc in ("utf-8-sig", "cp949", "euc-kr"):
        try:
            return pd.read_csv(path, dtype=str, encoding=enc)
        except UnicodeDecodeError:
            continue
    raise ValueError(f"인코딩을 판별할 수 없습니다: {path}")


def _norm(s):
    return re.sub(r"[\s_()\[\]%]", "", str(s)).lower()


def map_columns(df, alias_map, required=(), label="데이터"):
    """alias_map = {내부이름: [후보컬럼...]} 에 따라 컬럼명을 내부 이름으로 바꾼다.
    정확히 일치 → 공백·괄호 무시 일치 순서로 찾는다."""
    rename, found = {}, {}
    normcols = {_norm(c): c for c in df.columns}
    for key, cands in alias_map.items():
        hit = next((c for c in cands if c in df.columns), None)
        if hit is None:
            hit = next((normcols[_norm(c)] for c in cands if _norm(c) in normcols), None)
        if hit is not None:
            rename[hit] = key
            found[key] = hit
    missing = [k for k in required if k not in found]
    if missing:
        raise KeyError(
            f"{label}에서 필수 컬럼 {missing}을 찾지 못했습니다.\n"
            f"원자료 컬럼: {list(df.columns)}\n"
            f"config.py의 별칭 목록에 실제 컬럼명을 추가하세요."
        )
    print(f"[{label}] 컬럼 매핑: " + ", ".join(f"{k}←'{v}'" for k, v in found.items()))
    return df.rename(columns=rename)[list(found)]


def to_num(s):
    """'1,234', '1234㎡', '' 같은 값을 숫자로."""
    return pd.to_numeric(
        s.astype(str).str.replace(r"[^\d.\-]", "", regex=True).replace("", None),
        errors="coerce",
    )


def ksic2_of(code):
    """'C24111', '24111', '24' → 24. 숫자가 없으면 NaN."""
    m = re.search(r"(\d{2})", str(code))
    return int(m.group(1)) if m else None


def ksic_group(k2):
    for g, lst in KSIC2_GROUP.items():
        if k2 in lst:
            return g
    return None


def norm_complex_name(s):
    """단지명 매칭용 정규화: '반월국가산업단지' / '반월 국가산단' / '반월(국가)' → '반월'."""
    s = str(s)
    s = re.sub(r"\(.*?\)", "", s)
    s = re.sub(r"(국가|일반|도시첨단|농공)?\s*(산업)?(단지|산단)$", "", s.strip())
    s = re.sub(r"(국가|일반|도시첨단|농공)$", "", s.strip())
    return re.sub(r"\s+", "", s)
