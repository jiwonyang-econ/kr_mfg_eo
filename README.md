# Korean Manufacturing under the 2025–26 U.S. Tariff Regime

Data pipeline for an ongoing research project on how Korean manufacturers responded to, and recovered from, the 2025–26 U.S. tariff measures (Section 232 tariffs on autos, parts, steel, aluminum, copper, wood products and trucks; IEEPA "reciprocal" tariffs and their removal by the Supreme Court in February 2026; the Section 122 tariff).

The project combines four kinds of data:

| Source | Content | Unit |
|---|---|---|
| Korea Customs Service (via data.go.kr) | Exports by HS product × destination; exports by municipality (place of manufacture) × HS6 | monthly, 2018– |
| KICOX factory registration (via data.go.kr) | Registered manufacturing plants: address, industry (KSIC), products, employment, floor area | plant |
| Statistics Korea (KOSIS) | Mining and manufacturing production, shipment and inventory indices by province × industry | monthly, 2018– |
| Satellite imagery (Google Earth Engine) | VIIRS night-time lights (monthly composites and daily Black Marble), Sentinel-5P NO₂, Landsat surface temperature | plant site / industrial complex, monthly |

**Status:** work in progress (October 2026). Results are preliminary and not for citation.

**Author:** Jiwon Yang (PhD student in economics)

## Research questions

1. **Announcement and uncertainty vs. tariff level.** Do exports respond when tariffs are announced, when they take effect, or only to the rate itself? Do they recover when uncertainty is resolved (bilateral deal, court ruling) before or after the rate changes?
2. **Incidence.** Did Korean exporters absorb tariffs through lower export (FOB) prices to the U.S. relative to other destinations?
3. **Supplier spillovers.** How far does a shock to final-goods exporters travel to domestic suppliers in the same industrial clusters?
4. **Capacity reallocation.** Do firms with U.S. production shift capacity away from domestic plants, and is this visible in plant-level satellite data?

## Pipeline

| Step | Script | What it does |
|---|---|---|
| 0 | `00_fetch_api.py` | Factory registration and production data (KICOX APIs), paginated with resume |
| 0b | `00b_fetch_kicox.py` | Quarterly industrial-complex statistics (production, exports, utilization) |
| 0c | `00c_fetch_kosis.py` | KOSIS statistical tables (direct or via the data.go.kr gateway) |
| 0d | `00d_fetch_customs.py` | Customs exports by HS product × destination country, monthly |
| 0e | `00e_fetch_customs_sgg.py` | Customs exports by municipality × HS6, monthly; harmonizes province codes across the July 2026 Gwangju–South Jeolla merger |
| 1 | `01_clean_factories.py` | Restricts to manufacturing (KSIC 10–34), cleans addresses, flags large plants and multi-tenant/developer records |
| 2 | `02_geocode.py` | Geocodes plant addresses (Kakao Local API) with caching and quota handling |
| 3 | `03_build_units.py` | Builds analysis units: large-plant sites and industrial-complex polygons with industry composition |
| 4 | `04_gee_extract.py` | Extracts satellite indicators for each unit in Google Earth Engine (monthly or weekly) |
| 5 | `05_validate.py` | Validates satellite indicators against quarterly complex statistics |
| 6 | `06_validate_monthly.py` | Validates satellite indicators against monthly province × industry production indices |
| 7 | `07_tariff_event.py`, `07b_tariff_event_v2.py` | Exposure measures and event studies for the auto supply chain (product-level within municipality-month; municipality-level; policy-period pooled estimates with wild cluster bootstrap) |

Shared settings (paths, column aliases, industry groups, sample period) are in `config.py`; helpers in `utils.py`. `tests/make_sample_data.py` generates synthetic inputs for testing the pipeline without API access.

## Running

Python 3.10+.

```bash
pip install pandas numpy geopandas shapely pyproj requests statsmodels openpyxl matplotlib earthengine-api
```

API keys are read from environment variables and are never stored in the code:

```bash
export DATA_GO_KR_KEY=...   # data.go.kr service key
export KAKAO_KEY=...        # Kakao REST API key (geocoding)
export KOSIS_KEY=...        # optional, kosis.kr direct access
```

Example:

```bash
python 00d_fetch_customs.py get --cnty US --hs auto4 --start 201801 --end 202608
python 00e_fetch_customs_sgg.py get --hs auto --start 201801 --end 202608
python 07b_tariff_event_v2.py
```

Raw and processed data are not included in this repository (`data/` is excluded). All sources are public; see the table above.

## Preliminary findings

- Satellite night-time lights track the stock of industrial capacity (cross-section, annual changes; β ≈ 0.7–1.2 against complex production) but not month-to-month utilization (β ≈ 0 against monthly production indices). They are therefore used for capacity questions rather than short-run output.
- In the auto supply chain, products with a high U.S. share show no relative decline in vehicle export volumes, consistent with tariff absorption in margins, while parts show a modest decline concentrated in the 25 percent period.

## Data sources

- Korea Customs Service, *Export–import statistics by item and country* (data.go.kr 15100475) and *by municipality and item* (data.go.kr 15134343)
- Korea Industrial Complex Corporation, factory registration information and industrial-complex statistics (data.go.kr)
- Statistics Korea, *Mining and manufacturing production index by province and industry* (DT_1F02001)
- NOAA/NASA VIIRS night-time lights; Copernicus Sentinel-5P; USGS Landsat (via Google Earth Engine)
