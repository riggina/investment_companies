import csv
import io
import json
import os
import re
import time
from collections import deque
from datetime import datetime, timezone, timedelta
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import psycopg
import requests
import streamlit as st


# =========================================================
# SAFE SECRET READER
# =========================================================
def get_secret(key: str, default: Optional[str] = None) -> Optional[str]:
    try:
        if key in st.secrets:
            return st.secrets[key]
    except Exception:
        pass
    return os.getenv(key, default)


COMPANIES_HOUSE_API_KEY = (get_secret("COMPANIES_HOUSE_API_KEY", "") or "").strip()
SUPABASE_DB_URL = (get_secret("SUPABASE_DB_URL", "") or "").strip()
ADMIN_ACCESS_TOKEN = (get_secret("ADMIN_ACCESS_TOKEN", "") or "").strip()

CACHE_TTL_SEARCH_HOURS = int(get_secret("CACHE_TTL_SEARCH_HOURS", "24"))
CACHE_TTL_COMPANY_HOURS = int(get_secret("CACHE_TTL_COMPANY_HOURS", "168"))
CACHE_TTL_OFFICERS_HOURS = int(get_secret("CACHE_TTL_OFFICERS_HOURS", "168"))
SEARCH_RESULT_LIMIT = int(get_secret("SEARCH_RESULT_LIMIT", "10"))
API_MAX_REQUESTS_PER_5_MIN = int(get_secret("API_MAX_REQUESTS_PER_5_MIN", "540"))
ADMIN_BATCH_SIZE = int(get_secret("ADMIN_BATCH_SIZE", "100"))

BASE_URL = "https://api.company-information.service.gov.uk"


# =========================================================
# PAGE SETUP
# =========================================================
st.set_page_config(
    page_title="UK Investment Companies Lookup",
    page_icon="🏢",
    layout="wide",
)

st.markdown(
    """
    <style>
    .block-container {
        padding-top: 1.4rem;
        padding-bottom: 2rem;
        max-width: 1380px;
    }

    .hero {
        padding: 1.4rem 1.6rem;
        border-radius: 18px;
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 45%, #2563eb 100%);
        color: white;
        margin-bottom: 1rem;
        box-shadow: 0 10px 30px rgba(15, 23, 42, 0.18);
    }

    .hero h1 {
        margin: 0 0 0.35rem 0;
        font-size: 2rem;
        font-weight: 700;
    }

    .hero p {
        margin: 0;
        opacity: 0.92;
        font-size: 0.98rem;
    }

    .section-card {
        border: 1px solid rgba(120,120,120,0.15);
        border-radius: 16px;
        padding: 1rem 1rem 0.8rem 1rem;
        background: rgba(255,255,255,0.02);
        margin-bottom: 1rem;
    }

    .small-muted {
        color: #64748b;
        font-size: 0.9rem;
    }

    .score-box {
        padding: 1rem 1.2rem;
        border-radius: 16px;
        background: linear-gradient(135deg, rgba(37,99,235,.10), rgba(16,185,129,.10));
        border: 1px solid rgba(37,99,235,.18);
        margin-bottom: 1rem;
    }

    .label-pill {
        display: inline-block;
        padding: 0.3rem 0.6rem;
        border-radius: 999px;
        background: rgba(37,99,235,.10);
        border: 1px solid rgba(37,99,235,.18);
        font-size: 0.82rem;
        margin-right: 0.35rem;
        margin-bottom: 0.35rem;
    }

    .admin-banner {
        padding: .85rem 1rem;
        border-radius: 14px;
        background: linear-gradient(135deg, rgba(245,158,11,.15), rgba(234,88,12,.10));
        border: 1px solid rgba(245,158,11,.25);
        margin-bottom: 1rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# =========================================================
# HELPERS
# =========================================================
def utcnow() -> datetime:
    return datetime.now(timezone.utc)


def normalize_query(text: str) -> str:
    text = (text or "").strip().lower()
    text = re.sub(r"\s+", " ", text)
    return text


def parse_date(value: Optional[str]) -> Optional[str]:
    if not value:
        return None
    return value


def to_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, default=str)


def is_stale(dt: Optional[datetime], ttl_hours: int) -> bool:
    if dt is None:
        return True
    return utcnow() - dt > timedelta(hours=ttl_hours)


def format_date(value: Any) -> str:
    if value is None or value == "":
        return "-"
    return str(value)


def score_color(score: int) -> str:
    if score >= 45:
        return "🟢"
    if score >= 20:
        return "🟡"
    return "🔴"


def chunk_list(items: List[str], size: int) -> List[List[str]]:
    return [items[i:i + size] for i in range(0, len(items), size)]


def detect_keyword_column(df: pd.DataFrame) -> Optional[str]:
    preferred_names = ["keyword", "keywords", "company", "companies", "company_name", "query", "queries", "name"]
    normalized_cols = {str(c).strip().lower(): c for c in df.columns}

    for name in preferred_names:
        if name in normalized_cols:
            return normalized_cols[name]

    if len(df.columns) == 1:
        return df.columns[0]

    return None


# =========================================================
# CONFIG ERROR UI
# =========================================================
def show_config_error():
    st.error("Configuration is missing.")
    st.info(
        "Set these values in Streamlit secrets or environment variables:\n"
        "- COMPANIES_HOUSE_API_KEY\n"
        "- SUPABASE_DB_URL\n"
        "- ADMIN_ACCESS_TOKEN"
    )
    st.code(
        'COMPANIES_HOUSE_API_KEY = "your_key"\n'
        'SUPABASE_DB_URL = "postgresql://..."\n'
        'ADMIN_ACCESS_TOKEN = "your_admin_token"',
        language="toml"
    )


# =========================================================
# RATE LIMITER
# =========================================================
class RateLimiter:
    def __init__(self, max_requests: int, window_seconds: int = 300):
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self.request_times = deque()

    def wait_if_needed(self):
        now = time.time()

        while self.request_times and (now - self.request_times[0] >= self.window_seconds):
            self.request_times.popleft()

        if len(self.request_times) >= self.max_requests:
            sleep_for = self.window_seconds - (now - self.request_times[0]) + 0.2
            if sleep_for > 0:
                time.sleep(sleep_for)

            now = time.time()
            while self.request_times and (now - self.request_times[0] >= self.window_seconds):
                self.request_times.popleft()

        self.request_times.append(time.time())


@st.cache_resource
def get_rate_limiter():
    return RateLimiter(API_MAX_REQUESTS_PER_5_MIN, 300)


# =========================================================
# CONNECTIONS
# =========================================================
@st.cache_resource
def get_db_connection():
    return psycopg.connect(SUPABASE_DB_URL, autocommit=True)


@st.cache_resource
def get_http_session():
    session = requests.Session()
    session.auth = (COMPANIES_HOUSE_API_KEY, "")
    session.headers.update({
        "Accept": "application/json",
        "User-Agent": "Streamlit UK Investment Companies Lookup"
    })
    return session


# =========================================================
# DB FUNCTIONS
# =========================================================
def db_fetchone(query: str, params: Tuple = ()) -> Optional[tuple]:
    conn = get_db_connection()
    with conn.cursor() as cur:
        cur.execute(query, params)
        return cur.fetchone()


def db_fetchall(query: str, params: Tuple = ()) -> List[tuple]:
    conn = get_db_connection()
    with conn.cursor() as cur:
        cur.execute(query, params)
        return cur.fetchall()


def db_execute(query: str, params: Tuple = ()) -> None:
    conn = get_db_connection()
    with conn.cursor() as cur:
        cur.execute(query, params)


# =========================================================
# COMPANY TYPE TABLE / SIC MAPPING
# =========================================================
def ensure_company_type_table() -> None:
    """Creates the company_type lookup table and adds extra columns to companies if absent."""
    db_execute("""
        create table if not exists public.company_type (
            id          bigserial primary key,
            sic_code    text      not null unique,
            description text      not null,
            section     text      not null
        )
    """)
    # Safely add new columns to companies (ignored if they already exist)
    for col_def in [
        "alter table public.companies add column if not exists region       text",
        "alter table public.companies add column if not exists sic_code     text",
        "alter table public.companies add column if not exists section_type text",
    ]:
        try:
            db_execute(col_def)
        except Exception:
            pass


@st.cache_data(ttl=3600)
def lookup_company_type(sic_code: str) -> Tuple[Optional[str], Optional[str]]:
    """Returns (description, section) for a SIC code, or (None, None) if not found."""
    if not sic_code:
        return None, None
    row = db_fetchone(
        "select description, section from public.company_type where sic_code = %s",
        (str(sic_code),)
    )
    if row:
        return row[0], row[1]
    return None, None


@st.cache_data(ttl=300)
def get_all_regions() -> List[str]:
    rows = db_fetchall(
        "select distinct region from public.companies where region is not null order by region"
    )
    return [r[0] for r in rows]


@st.cache_data(ttl=300)
def get_all_sections() -> List[str]:
    rows = db_fetchall(
        "select distinct section from public.company_type where section is not null order by section"
    )
    return [r[0] for r in rows]


def seed_company_type_table() -> int:
    """Seeds standard UK SIC 2007 codes into the company_type table. Returns count inserted."""
    SIC_DATA = [
        # Section A – Agriculture, Forestry and Fishing
        ("01110","Growing of cereals (except rice), leguminous crops and oil seeds","A – Agriculture, Forestry and Fishing"),
        ("01120","Growing of rice","A – Agriculture, Forestry and Fishing"),
        ("01130","Growing of vegetables and melons, roots and tubers","A – Agriculture, Forestry and Fishing"),
        ("01210","Growing of grapes","A – Agriculture, Forestry and Fishing"),
        ("01220","Growing of tropical and subtropical fruits","A – Agriculture, Forestry and Fishing"),
        ("01230","Growing of citrus fruits","A – Agriculture, Forestry and Fishing"),
        ("01250","Growing of other tree and bush fruits and nuts","A – Agriculture, Forestry and Fishing"),
        ("01300","Plant propagation","A – Agriculture, Forestry and Fishing"),
        ("01410","Raising of dairy cattle","A – Agriculture, Forestry and Fishing"),
        ("01420","Raising of other cattle and buffaloes","A – Agriculture, Forestry and Fishing"),
        ("01450","Raising of sheep and goats","A – Agriculture, Forestry and Fishing"),
        ("01460","Raising of swine/pigs","A – Agriculture, Forestry and Fishing"),
        ("01470","Raising of poultry","A – Agriculture, Forestry and Fishing"),
        ("01500","Mixed farming","A – Agriculture, Forestry and Fishing"),
        ("01610","Support activities for crop production","A – Agriculture, Forestry and Fishing"),
        ("01700","Hunting, trapping and related service activities","A – Agriculture, Forestry and Fishing"),
        ("02100","Silviculture and other forestry activities","A – Agriculture, Forestry and Fishing"),
        ("02200","Logging","A – Agriculture, Forestry and Fishing"),
        ("02300","Gathering of wild growing non-wood products","A – Agriculture, Forestry and Fishing"),
        ("02400","Support services to forestry","A – Agriculture, Forestry and Fishing"),
        ("03110","Marine fishing","A – Agriculture, Forestry and Fishing"),
        ("03120","Freshwater fishing","A – Agriculture, Forestry and Fishing"),
        ("03210","Marine aquaculture","A – Agriculture, Forestry and Fishing"),
        ("03220","Freshwater aquaculture","A – Agriculture, Forestry and Fishing"),
        # Section B – Mining and Quarrying
        ("05101","Deep coal mines","B – Mining and Quarrying"),
        ("05102","Open cast coal working","B – Mining and Quarrying"),
        ("06100","Extraction of crude petroleum","B – Mining and Quarrying"),
        ("06200","Extraction of natural gas","B – Mining and Quarrying"),
        ("07100","Mining of iron ores","B – Mining and Quarrying"),
        ("07290","Mining of other non-ferrous metal ores","B – Mining and Quarrying"),
        ("08110","Quarrying of ornamental and building stone","B – Mining and Quarrying"),
        ("08120","Operation of gravel and sand pits","B – Mining and Quarrying"),
        ("09100","Support activities for petroleum and natural gas extraction","B – Mining and Quarrying"),
        # Section C – Manufacturing
        ("10110","Processing and preserving of meat","C – Manufacturing"),
        ("10120","Processing and preserving of poultry meat","C – Manufacturing"),
        ("10130","Production of meat and poultry meat products","C – Manufacturing"),
        ("10200","Processing and preserving of fish","C – Manufacturing"),
        ("10310","Processing and preserving of potatoes","C – Manufacturing"),
        ("10320","Manufacture of fruit and vegetable juice","C – Manufacturing"),
        ("10410","Manufacture of oils and fats","C – Manufacturing"),
        ("10511","Liquid milk and cream production","C – Manufacturing"),
        ("10512","Butter and cheese production","C – Manufacturing"),
        ("10611","Grain milling","C – Manufacturing"),
        ("10620","Manufacture of starches and starch products","C – Manufacturing"),
        ("10710","Manufacture of bread; manufacture of fresh pastry goods and cakes","C – Manufacturing"),
        ("10720","Manufacture of rusks and biscuits","C – Manufacturing"),
        ("10810","Manufacture of sugar","C – Manufacturing"),
        ("10821","Manufacture of cocoa and chocolate confectionery","C – Manufacturing"),
        ("10910","Manufacture of prepared feeds for farm animals","C – Manufacturing"),
        ("11010","Distilling, rectifying and blending of spirits","C – Manufacturing"),
        ("11020","Manufacture of wine from grape","C – Manufacturing"),
        ("11050","Manufacture of beer","C – Manufacturing"),
        ("11060","Manufacture of malt","C – Manufacturing"),
        ("11070","Manufacture of soft drinks","C – Manufacturing"),
        ("12000","Manufacture of tobacco products","C – Manufacturing"),
        ("13100","Preparation and spinning of textile fibres","C – Manufacturing"),
        ("13200","Weaving of textiles","C – Manufacturing"),
        ("13300","Finishing of textiles","C – Manufacturing"),
        ("13910","Manufacture of knitted and crocheted fabrics","C – Manufacturing"),
        ("14110","Manufacture of leather clothes","C – Manufacturing"),
        ("14120","Manufacture of workwear","C – Manufacturing"),
        ("14190","Manufacture of other wearing apparel and accessories","C – Manufacturing"),
        ("14200","Manufacture of articles of fur","C – Manufacturing"),
        ("14310","Manufacture of knitted and crocheted hosiery","C – Manufacturing"),
        ("15110","Tanning and dressing of leather","C – Manufacturing"),
        ("15120","Manufacture of luggage, handbags and the like","C – Manufacturing"),
        ("15200","Manufacture of footwear","C – Manufacturing"),
        ("16100","Sawmilling and planing of wood","C – Manufacturing"),
        ("16210","Manufacture of veneer sheets and wood-based panels","C – Manufacturing"),
        ("16230","Manufacture of other builders carpentry and joinery","C – Manufacturing"),
        ("16240","Manufacture of wooden containers","C – Manufacturing"),
        ("17110","Manufacture of pulp","C – Manufacturing"),
        ("17120","Manufacture of paper and paperboard","C – Manufacturing"),
        ("17210","Manufacture of corrugated paper","C – Manufacturing"),
        ("17290","Manufacture of other articles of paper and paperboard","C – Manufacturing"),
        ("17300","Manufacture of articles of stationery","C – Manufacturing"),
        ("18110","Printing of newspapers","C – Manufacturing"),
        ("18121","Manufacture of printed labels","C – Manufacturing"),
        ("18129","Printing (other than printing of newspapers) n.e.c.","C – Manufacturing"),
        ("18130","Pre-press and pre-media services","C – Manufacturing"),
        ("18140","Binding and related services","C – Manufacturing"),
        ("18200","Reproduction of recorded media","C – Manufacturing"),
        ("19100","Manufacture of coke oven products","C – Manufacturing"),
        ("19201","Mineral oil refining","C – Manufacturing"),
        ("20110","Manufacture of industrial gases","C – Manufacturing"),
        ("20120","Manufacture of dyes and pigments","C – Manufacturing"),
        ("20130","Manufacture of other inorganic basic chemicals","C – Manufacturing"),
        ("20140","Manufacture of other organic basic chemicals","C – Manufacturing"),
        ("20150","Manufacture of fertilisers and nitrogen compounds","C – Manufacturing"),
        ("20160","Manufacture of plastics in primary forms","C – Manufacturing"),
        ("20170","Manufacture of synthetic rubber in primary forms","C – Manufacturing"),
        ("20200","Manufacture of pesticides and other agrochemical products","C – Manufacturing"),
        ("20301","Manufacture of paints, varnishes and similar coatings","C – Manufacturing"),
        ("20302","Manufacture of printing ink","C – Manufacturing"),
        ("20410","Manufacture of soap and detergents","C – Manufacturing"),
        ("20420","Manufacture of perfumes and toilet preparations","C – Manufacturing"),
        ("20510","Manufacture of explosives","C – Manufacturing"),
        ("20520","Manufacture of glues","C – Manufacturing"),
        ("20600","Manufacture of man-made fibres","C – Manufacturing"),
        ("21100","Manufacture of basic pharmaceutical products","C – Manufacturing"),
        ("21200","Manufacture of pharmaceutical preparations","C – Manufacturing"),
        ("22110","Manufacture of rubber tyres and tubes","C – Manufacturing"),
        ("22190","Manufacture of other rubber products","C – Manufacturing"),
        ("22210","Manufacture of plastic plates, sheets, tubes and profiles","C – Manufacturing"),
        ("22220","Manufacture of plastic packing goods","C – Manufacturing"),
        ("22230","Manufacture of builders ware of plastic","C – Manufacturing"),
        ("22290","Manufacture of other plastic products","C – Manufacturing"),
        ("23110","Manufacture of flat glass","C – Manufacturing"),
        ("23200","Manufacture of refractory products","C – Manufacturing"),
        ("23300","Manufacture of clay building materials","C – Manufacturing"),
        ("23410","Manufacture of ceramic household and ornamental articles","C – Manufacturing"),
        ("23510","Manufacture of cement","C – Manufacturing"),
        ("23520","Manufacture of lime and plaster","C – Manufacturing"),
        ("23610","Manufacture of concrete products for construction purposes","C – Manufacturing"),
        ("23700","Cutting, shaping and finishing of stone","C – Manufacturing"),
        ("23910","Production of abrasive products","C – Manufacturing"),
        ("24100","Manufacture of basic iron and steel","C – Manufacturing"),
        ("24200","Manufacture of tubes, pipes, hollow profiles","C – Manufacturing"),
        ("24310","Cold drawing of bars","C – Manufacturing"),
        ("24410","Precious metals production","C – Manufacturing"),
        ("24420","Aluminium production","C – Manufacturing"),
        ("24510","Casting of iron","C – Manufacturing"),
        ("25110","Manufacture of metal structures and parts","C – Manufacturing"),
        ("25120","Manufacture of doors and windows of metal","C – Manufacturing"),
        ("25210","Manufacture of central heating radiators","C – Manufacturing"),
        ("25300","Manufacture of steam generators","C – Manufacturing"),
        ("25400","Manufacture of weapons and ammunition","C – Manufacturing"),
        ("25500","Forging, pressing, stamping of metal","C – Manufacturing"),
        ("25610","Treatment and coating of metals","C – Manufacturing"),
        ("25620","Machining","C – Manufacturing"),
        ("25710","Manufacture of cutlery","C – Manufacturing"),
        ("25720","Manufacture of locks and hinges","C – Manufacturing"),
        ("25730","Manufacture of tools","C – Manufacturing"),
        ("25910","Manufacture of steel drums and similar containers","C – Manufacturing"),
        ("25940","Manufacture of fasteners and screw machine products","C – Manufacturing"),
        ("25990","Manufacture of other fabricated metal products n.e.c.","C – Manufacturing"),
        ("26110","Manufacture of electronic components","C – Manufacturing"),
        ("26120","Manufacture of loaded electronic boards","C – Manufacturing"),
        ("26200","Manufacture of computers and peripheral equipment","C – Manufacturing"),
        ("26301","Manufacture of telegraph and telephone apparatus","C – Manufacturing"),
        ("26309","Manufacture of communication equipment (other)","C – Manufacturing"),
        ("26400","Manufacture of consumer electronics","C – Manufacturing"),
        ("26511","Manufacture of electronic instruments for measuring","C – Manufacturing"),
        ("26512","Manufacture of non-electronic instruments for measuring","C – Manufacturing"),
        ("26520","Manufacture of watches and clocks","C – Manufacturing"),
        ("26600","Manufacture of irradiation equipment","C – Manufacturing"),
        ("26700","Manufacture of optical instruments and photographic equipment","C – Manufacturing"),
        ("26800","Manufacture of magnetic and optical media","C – Manufacturing"),
        ("27110","Manufacture of electric motors, generators and transformers","C – Manufacturing"),
        ("27120","Manufacture of electricity distribution and control apparatus","C – Manufacturing"),
        ("27200","Manufacture of batteries and accumulators","C – Manufacturing"),
        ("27310","Manufacture of fibre optic cables","C – Manufacturing"),
        ("27320","Manufacture of other electronic and electric wires and cables","C – Manufacturing"),
        ("27330","Manufacture of wiring devices","C – Manufacturing"),
        ("27400","Manufacture of electric lighting equipment","C – Manufacturing"),
        ("27510","Manufacture of electric domestic appliances","C – Manufacturing"),
        ("27520","Manufacture of non-electric domestic appliances","C – Manufacturing"),
        ("27900","Manufacture of other electrical equipment","C – Manufacturing"),
        ("28110","Manufacture of engines and turbines","C – Manufacturing"),
        ("28120","Manufacture of fluid power equipment","C – Manufacturing"),
        ("28130","Manufacture of other pumps and compressors","C – Manufacturing"),
        ("28140","Manufacture of other taps and valves","C – Manufacturing"),
        ("28150","Manufacture of bearings, gears, gearing and driving elements","C – Manufacturing"),
        ("28210","Manufacture of ovens, furnaces and furnace burners","C – Manufacturing"),
        ("28220","Manufacture of lifting and handling equipment","C – Manufacturing"),
        ("28230","Manufacture of office machinery and equipment","C – Manufacturing"),
        ("28240","Manufacture of power-driven hand tools","C – Manufacturing"),
        ("28250","Manufacture of non-domestic cooling and ventilation equipment","C – Manufacturing"),
        ("28290","Manufacture of other general-purpose machinery n.e.c.","C – Manufacturing"),
        ("28300","Manufacture of agricultural and forestry machinery","C – Manufacturing"),
        ("28410","Manufacture of metal forming machinery","C – Manufacturing"),
        ("28490","Manufacture of other machine tools","C – Manufacturing"),
        ("28910","Manufacture of machinery for metallurgy","C – Manufacturing"),
        ("28920","Manufacture of machinery for mining","C – Manufacturing"),
        ("28930","Manufacture of machinery for food and beverage processing","C – Manufacturing"),
        ("28940","Manufacture of machinery for textile production","C – Manufacturing"),
        ("28950","Manufacture of machinery for paper and paperboard production","C – Manufacturing"),
        ("28960","Manufacture of plastics and rubber machinery","C – Manufacturing"),
        ("28990","Manufacture of other special-purpose machinery n.e.c.","C – Manufacturing"),
        ("29100","Manufacture of motor vehicles","C – Manufacturing"),
        ("29200","Manufacture of bodies for motor vehicles","C – Manufacturing"),
        ("29310","Manufacture of electrical and electronic equipment for motor vehicles","C – Manufacturing"),
        ("29320","Manufacture of other parts and accessories for motor vehicles","C – Manufacturing"),
        ("30110","Building of ships and floating structures","C – Manufacturing"),
        ("30120","Building of pleasure and sporting boats","C – Manufacturing"),
        ("30200","Manufacture of railway locomotives and rolling stock","C – Manufacturing"),
        ("30300","Manufacture of air and spacecraft and related machinery","C – Manufacturing"),
        ("30400","Manufacture of military fighting vehicles","C – Manufacturing"),
        ("30910","Manufacture of motorcycles","C – Manufacturing"),
        ("30920","Manufacture of bicycles and invalid carriages","C – Manufacturing"),
        ("30990","Manufacture of other transport equipment n.e.c.","C – Manufacturing"),
        ("31010","Manufacture of office and shop furniture","C – Manufacturing"),
        ("31020","Manufacture of kitchen furniture","C – Manufacturing"),
        ("31030","Manufacture of mattresses","C – Manufacturing"),
        ("31090","Manufacture of other furniture","C – Manufacturing"),
        ("32110","Striking of coins","C – Manufacturing"),
        ("32120","Manufacture of jewellery and related articles","C – Manufacturing"),
        ("32130","Manufacture of imitation jewellery and related articles","C – Manufacturing"),
        ("32200","Manufacture of musical instruments","C – Manufacturing"),
        ("32300","Manufacture of sports goods","C – Manufacturing"),
        ("32400","Manufacture of games and toys","C – Manufacturing"),
        ("32500","Manufacture of medical and dental instruments and supplies","C – Manufacturing"),
        ("32910","Manufacture of brooms and brushes","C – Manufacturing"),
        ("32990","Other manufacturing n.e.c.","C – Manufacturing"),
        ("33110","Repair of fabricated metal products","C – Manufacturing"),
        ("33120","Repair of machinery","C – Manufacturing"),
        ("33130","Repair of electronic and optical equipment","C – Manufacturing"),
        ("33140","Repair of electrical equipment","C – Manufacturing"),
        ("33150","Repair and maintenance of ships and boats","C – Manufacturing"),
        ("33160","Repair and maintenance of aircraft and spacecraft","C – Manufacturing"),
        ("33170","Repair and maintenance of other transport equipment","C – Manufacturing"),
        ("33190","Repair of other equipment","C – Manufacturing"),
        ("33200","Installation of industrial machinery and equipment","C – Manufacturing"),
        # Section D – Electricity, Gas, Steam and Air Conditioning Supply
        ("35110","Production of electricity","D – Electricity, Gas, Steam and Air Conditioning Supply"),
        ("35120","Transmission of electricity","D – Electricity, Gas, Steam and Air Conditioning Supply"),
        ("35130","Distribution of electricity","D – Electricity, Gas, Steam and Air Conditioning Supply"),
        ("35140","Trade of electricity","D – Electricity, Gas, Steam and Air Conditioning Supply"),
        ("35210","Manufacture of gas","D – Electricity, Gas, Steam and Air Conditioning Supply"),
        ("35220","Distribution of gaseous fuels through mains","D – Electricity, Gas, Steam and Air Conditioning Supply"),
        ("35230","Trade of gas through mains","D – Electricity, Gas, Steam and Air Conditioning Supply"),
        ("35300","Steam and air conditioning supply","D – Electricity, Gas, Steam and Air Conditioning Supply"),
        # Section E – Water Supply; Sewerage, Waste Management
        ("36000","Water collection, treatment and supply","E – Water Supply; Sewerage, Waste Management"),
        ("37000","Sewerage","E – Water Supply; Sewerage, Waste Management"),
        ("38110","Collection of non-hazardous waste","E – Water Supply; Sewerage, Waste Management"),
        ("38120","Collection of hazardous waste","E – Water Supply; Sewerage, Waste Management"),
        ("38210","Treatment and disposal of non-hazardous waste","E – Water Supply; Sewerage, Waste Management"),
        ("38220","Treatment and disposal of hazardous waste","E – Water Supply; Sewerage, Waste Management"),
        ("38310","Dismantling of wrecks","E – Water Supply; Sewerage, Waste Management"),
        ("38320","Recovery of sorted materials","E – Water Supply; Sewerage, Waste Management"),
        ("39000","Remediation activities and other waste management services","E – Water Supply; Sewerage, Waste Management"),
        # Section F – Construction
        ("41100","Development of building projects","F – Construction"),
        ("41201","Construction of commercial buildings","F – Construction"),
        ("41202","Construction of domestic buildings","F – Construction"),
        ("42110","Construction of roads and motorways","F – Construction"),
        ("42120","Construction of railways and underground railways","F – Construction"),
        ("42130","Construction of bridges and tunnels","F – Construction"),
        ("42210","Construction of utility projects for fluids","F – Construction"),
        ("42220","Construction of utility projects for electricity and telecommunications","F – Construction"),
        ("42910","Construction of water projects","F – Construction"),
        ("42990","Construction of other civil engineering projects n.e.c.","F – Construction"),
        ("43110","Demolition","F – Construction"),
        ("43120","Site preparation","F – Construction"),
        ("43130","Test drilling and boring","F – Construction"),
        ("43210","Electrical installation","F – Construction"),
        ("43220","Plumbing, heat and air-conditioning installation","F – Construction"),
        ("43290","Other construction installation","F – Construction"),
        ("43310","Plastering","F – Construction"),
        ("43320","Joinery installation","F – Construction"),
        ("43330","Floor and wall covering","F – Construction"),
        ("43341","Painting","F – Construction"),
        ("43342","Glazing","F – Construction"),
        ("43390","Other building completion and finishing","F – Construction"),
        ("43910","Roofing activities","F – Construction"),
        ("43991","Scaffold erection","F – Construction"),
        ("43999","Specialised construction activities n.e.c.","F – Construction"),
        # Section G – Wholesale and Retail Trade
        ("45111","Sale of new cars and light motor vehicles","G – Wholesale and Retail Trade"),
        ("45112","Sale of used cars and light motor vehicles","G – Wholesale and Retail Trade"),
        ("45200","Maintenance and repair of motor vehicles","G – Wholesale and Retail Trade"),
        ("45310","Wholesale trade of motor vehicle parts and accessories","G – Wholesale and Retail Trade"),
        ("45320","Retail trade of motor vehicle parts and accessories","G – Wholesale and Retail Trade"),
        ("45400","Sale, maintenance and repair of motorcycles","G – Wholesale and Retail Trade"),
        ("46110","Agents selling agricultural raw materials","G – Wholesale and Retail Trade"),
        ("46120","Agents in the sale of fuels","G – Wholesale and Retail Trade"),
        ("46130","Agents in the sale of timber and building materials","G – Wholesale and Retail Trade"),
        ("46140","Agents in the sale of machinery","G – Wholesale and Retail Trade"),
        ("46150","Agents in the sale of furniture","G – Wholesale and Retail Trade"),
        ("46160","Agents in the sale of textiles, clothing and footwear","G – Wholesale and Retail Trade"),
        ("46170","Agents in the sale of food","G – Wholesale and Retail Trade"),
        ("46180","Agents in the sale of other products","G – Wholesale and Retail Trade"),
        ("46190","Agents in the sale of a variety of goods","G – Wholesale and Retail Trade"),
        ("46210","Wholesale of grain, unmanufactured tobacco, seeds","G – Wholesale and Retail Trade"),
        ("46220","Wholesale of flowers and plants","G – Wholesale and Retail Trade"),
        ("46230","Wholesale of live animals","G – Wholesale and Retail Trade"),
        ("46240","Wholesale of hides, skins and leather","G – Wholesale and Retail Trade"),
        ("46310","Wholesale of fruit and vegetables","G – Wholesale and Retail Trade"),
        ("46320","Wholesale of meat and meat products","G – Wholesale and Retail Trade"),
        ("46330","Wholesale of dairy products, eggs and edible oils","G – Wholesale and Retail Trade"),
        ("46340","Wholesale of beverages","G – Wholesale and Retail Trade"),
        ("46350","Wholesale of tobacco products","G – Wholesale and Retail Trade"),
        ("46360","Wholesale of sugar and chocolate","G – Wholesale and Retail Trade"),
        ("46370","Wholesale of coffee, tea, cocoa and spices","G – Wholesale and Retail Trade"),
        ("46380","Wholesale of other food","G – Wholesale and Retail Trade"),
        ("46390","Non-specialised wholesale of food","G – Wholesale and Retail Trade"),
        ("46410","Wholesale of textiles","G – Wholesale and Retail Trade"),
        ("46420","Wholesale of clothing and footwear","G – Wholesale and Retail Trade"),
        ("46430","Wholesale of electrical household appliances","G – Wholesale and Retail Trade"),
        ("46440","Wholesale of china and glassware","G – Wholesale and Retail Trade"),
        ("46450","Wholesale of perfume and cosmetics","G – Wholesale and Retail Trade"),
        ("46460","Wholesale of pharmaceutical goods","G – Wholesale and Retail Trade"),
        ("46470","Wholesale of furniture, carpets and lighting equipment","G – Wholesale and Retail Trade"),
        ("46480","Wholesale of watches and jewellery","G – Wholesale and Retail Trade"),
        ("46490","Wholesale of other household goods","G – Wholesale and Retail Trade"),
        ("46510","Wholesale of computers and computer peripheral equipment","G – Wholesale and Retail Trade"),
        ("46520","Wholesale of electronic and telecommunications equipment","G – Wholesale and Retail Trade"),
        ("46610","Wholesale of agricultural machinery","G – Wholesale and Retail Trade"),
        ("46620","Wholesale of machine tools","G – Wholesale and Retail Trade"),
        ("46630","Wholesale of mining, construction and civil engineering machinery","G – Wholesale and Retail Trade"),
        ("46640","Wholesale of machinery for the textile industry","G – Wholesale and Retail Trade"),
        ("46650","Wholesale of office furniture","G – Wholesale and Retail Trade"),
        ("46660","Wholesale of other office machinery and equipment","G – Wholesale and Retail Trade"),
        ("46690","Wholesale of other machinery and equipment","G – Wholesale and Retail Trade"),
        ("46711","Wholesale of petroleum and petroleum products","G – Wholesale and Retail Trade"),
        ("46720","Wholesale of metals and metal ores","G – Wholesale and Retail Trade"),
        ("46730","Wholesale of wood, construction materials and sanitary equipment","G – Wholesale and Retail Trade"),
        ("46740","Wholesale of hardware, plumbing and heating equipment","G – Wholesale and Retail Trade"),
        ("46750","Wholesale of chemical products","G – Wholesale and Retail Trade"),
        ("46760","Wholesale of other intermediate products","G – Wholesale and Retail Trade"),
        ("46770","Wholesale of waste and scrap","G – Wholesale and Retail Trade"),
        ("46900","Non-specialised wholesale trade","G – Wholesale and Retail Trade"),
        ("47110","Retail sale in non-specialised stores with food","G – Wholesale and Retail Trade"),
        ("47190","Other retail sale in non-specialised stores","G – Wholesale and Retail Trade"),
        ("47210","Retail sale of fruit and vegetables","G – Wholesale and Retail Trade"),
        ("47220","Retail sale of meat and meat products","G – Wholesale and Retail Trade"),
        ("47230","Retail sale of fish","G – Wholesale and Retail Trade"),
        ("47240","Retail sale of bread, cakes, flour confectionery and sugar","G – Wholesale and Retail Trade"),
        ("47250","Retail sale of beverages in specialised stores","G – Wholesale and Retail Trade"),
        ("47260","Retail sale of tobacco products in specialised stores","G – Wholesale and Retail Trade"),
        ("47290","Other retail sale of food in specialised stores","G – Wholesale and Retail Trade"),
        ("47300","Retail sale of automotive fuel in specialised stores","G – Wholesale and Retail Trade"),
        ("47410","Retail sale of computers and peripheral equipment","G – Wholesale and Retail Trade"),
        ("47420","Retail sale of telecommunications equipment","G – Wholesale and Retail Trade"),
        ("47430","Retail sale of audio and video equipment","G – Wholesale and Retail Trade"),
        ("47510","Retail sale of textiles in specialised stores","G – Wholesale and Retail Trade"),
        ("47520","Retail sale of hardware, paints and glass","G – Wholesale and Retail Trade"),
        ("47530","Retail sale of carpets, rugs, wall and floor coverings","G – Wholesale and Retail Trade"),
        ("47540","Retail sale of electrical household appliances","G – Wholesale and Retail Trade"),
        ("47590","Retail sale of furniture, lighting equipment and other household articles","G – Wholesale and Retail Trade"),
        ("47610","Retail sale of books in specialised stores","G – Wholesale and Retail Trade"),
        ("47620","Retail sale of newspapers and stationery","G – Wholesale and Retail Trade"),
        ("47630","Retail sale of music and video recordings","G – Wholesale and Retail Trade"),
        ("47640","Retail sale of sporting equipment","G – Wholesale and Retail Trade"),
        ("47650","Retail sale of games and toys","G – Wholesale and Retail Trade"),
        ("47710","Retail sale of clothing in specialised stores","G – Wholesale and Retail Trade"),
        ("47720","Retail sale of footwear and leather goods","G – Wholesale and Retail Trade"),
        ("47730","Dispensing chemist in specialised stores","G – Wholesale and Retail Trade"),
        ("47740","Retail sale of medical and orthopaedic goods","G – Wholesale and Retail Trade"),
        ("47750","Retail sale of cosmetic and toilet articles","G – Wholesale and Retail Trade"),
        ("47760","Retail sale of flowers, plants and seeds","G – Wholesale and Retail Trade"),
        ("47770","Retail sale of watches and jewellery","G – Wholesale and Retail Trade"),
        ("47781","Retail sale in commercial art galleries","G – Wholesale and Retail Trade"),
        ("47782","Retail sale by opticians","G – Wholesale and Retail Trade"),
        ("47789","Other retail sale of new goods in specialised stores","G – Wholesale and Retail Trade"),
        ("47791","Retail sale of antiques","G – Wholesale and Retail Trade"),
        ("47799","Other retail sale of second-hand goods","G – Wholesale and Retail Trade"),
        ("47810","Retail sale via stalls and markets of food","G – Wholesale and Retail Trade"),
        ("47820","Retail sale via stalls and markets of textiles and clothing","G – Wholesale and Retail Trade"),
        ("47890","Retail sale via stalls and markets of other goods","G – Wholesale and Retail Trade"),
        ("47910","Retail sale via mail order houses or via Internet","G – Wholesale and Retail Trade"),
        ("47990","Other retail sale not in stores, stalls or markets","G – Wholesale and Retail Trade"),
        # Section H – Transportation and Storage
        ("49100","Passenger rail transport, interurban","H – Transportation and Storage"),
        ("49200","Freight rail transport","H – Transportation and Storage"),
        ("49310","Urban and suburban passenger land transport","H – Transportation and Storage"),
        ("49320","Taxi operation","H – Transportation and Storage"),
        ("49390","Other passenger land transport n.e.c.","H – Transportation and Storage"),
        ("49410","Freight transport by road","H – Transportation and Storage"),
        ("49420","Removal services","H – Transportation and Storage"),
        ("49500","Transport via pipeline","H – Transportation and Storage"),
        ("50100","Sea and coastal passenger water transport","H – Transportation and Storage"),
        ("50200","Sea and coastal freight water transport","H – Transportation and Storage"),
        ("50300","Inland passenger water transport","H – Transportation and Storage"),
        ("50400","Inland freight water transport","H – Transportation and Storage"),
        ("51100","Passenger air transport","H – Transportation and Storage"),
        ("51210","Freight air transport","H – Transportation and Storage"),
        ("51220","Space transport","H – Transportation and Storage"),
        ("52100","Warehousing and storage","H – Transportation and Storage"),
        ("52210","Service activities incidental to land transportation","H – Transportation and Storage"),
        ("52220","Service activities incidental to water transportation","H – Transportation and Storage"),
        ("52230","Service activities incidental to air transportation","H – Transportation and Storage"),
        ("52240","Cargo handling","H – Transportation and Storage"),
        ("52290","Other supporting transport activities","H – Transportation and Storage"),
        ("53100","Postal activities under universal service obligation","H – Transportation and Storage"),
        ("53200","Other postal and courier activities","H – Transportation and Storage"),
        # Section I – Accommodation and Food Service
        ("55100","Hotels and similar accommodation","I – Accommodation and Food Service"),
        ("55201","Holiday centres and villages","I – Accommodation and Food Service"),
        ("55202","Youth hostels","I – Accommodation and Food Service"),
        ("55209","Other holiday and other short-stay accommodation","I – Accommodation and Food Service"),
        ("55300","Recreational vehicle parks, trailer parks and camping grounds","I – Accommodation and Food Service"),
        ("55900","Other accommodation","I – Accommodation and Food Service"),
        ("56101","Licensed restaurants","I – Accommodation and Food Service"),
        ("56102","Unlicensed restaurants and cafes","I – Accommodation and Food Service"),
        ("56103","Take-away food shops and mobile food stands","I – Accommodation and Food Service"),
        ("56210","Event catering activities","I – Accommodation and Food Service"),
        ("56290","Other food service activities","I – Accommodation and Food Service"),
        ("56301","Licensed clubs","I – Accommodation and Food Service"),
        ("56302","Public houses and bars","I – Accommodation and Food Service"),
        # Section J – Information and Communication
        ("58110","Book publishing","J – Information and Communication"),
        ("58120","Publishing of directories and mailing lists","J – Information and Communication"),
        ("58130","Publishing of newspapers","J – Information and Communication"),
        ("58141","Publishing of learned journals","J – Information and Communication"),
        ("58142","Publishing of consumer and business journals and periodicals","J – Information and Communication"),
        ("58190","Other publishing activities","J – Information and Communication"),
        ("58210","Publishing of computer games","J – Information and Communication"),
        ("58290","Other software publishing","J – Information and Communication"),
        ("59111","Motion picture production activities","J – Information and Communication"),
        ("59112","Video production activities","J – Information and Communication"),
        ("59113","Television programme production activities","J – Information and Communication"),
        ("59120","Motion picture, video and television programme post-production activities","J – Information and Communication"),
        ("59130","Motion picture, video and television programme distribution activities","J – Information and Communication"),
        ("59140","Motion picture projection activities","J – Information and Communication"),
        ("59200","Sound recording and music publishing activities","J – Information and Communication"),
        ("60100","Radio broadcasting","J – Information and Communication"),
        ("60200","Television programming and broadcasting activities","J – Information and Communication"),
        ("61100","Wired telecommunications activities","J – Information and Communication"),
        ("61200","Wireless telecommunications activities","J – Information and Communication"),
        ("61300","Satellite telecommunications activities","J – Information and Communication"),
        ("61900","Other telecommunications activities","J – Information and Communication"),
        ("62011","Ready-made interactive leisure and entertainment software development","J – Information and Communication"),
        ("62012","Business and domestic software development","J – Information and Communication"),
        ("62020","Information technology consultancy activities","J – Information and Communication"),
        ("62030","Computer facilities management activities","J – Information and Communication"),
        ("62090","Other information technology service activities","J – Information and Communication"),
        ("63110","Data processing, hosting and related activities","J – Information and Communication"),
        ("63120","Web portals","J – Information and Communication"),
        ("63910","News agency activities","J – Information and Communication"),
        ("63990","Other information service activities n.e.c.","J – Information and Communication"),
        # Section K – Financial and Insurance Activities
        ("64110","Central banking","K – Financial and Insurance Activities"),
        ("64191","Banks","K – Financial and Insurance Activities"),
        ("64192","Building societies","K – Financial and Insurance Activities"),
        ("64201","Activities of agricultural holding companies","K – Financial and Insurance Activities"),
        ("64202","Activities of production holding companies","K – Financial and Insurance Activities"),
        ("64203","Activities of construction holding companies","K – Financial and Insurance Activities"),
        ("64204","Activities of distribution holding companies","K – Financial and Insurance Activities"),
        ("64205","Activities of financial services holding companies","K – Financial and Insurance Activities"),
        ("64209","Activities of other holding companies n.e.c.","K – Financial and Insurance Activities"),
        ("64301","Activities of investment trusts","K – Financial and Insurance Activities"),
        ("64302","Activities of unit trusts","K – Financial and Insurance Activities"),
        ("64303","Activities of venture and development capital companies","K – Financial and Insurance Activities"),
        ("64304","Activities of open-ended investment companies","K – Financial and Insurance Activities"),
        ("64305","Activities of real estate investment trusts","K – Financial and Insurance Activities"),
        ("64306","Activities of venture capital companies","K – Financial and Insurance Activities"),
        ("64910","Financial leasing","K – Financial and Insurance Activities"),
        ("64921","Credit granting by non-deposit taking finance houses","K – Financial and Insurance Activities"),
        ("64922","Activities of mortgage finance companies","K – Financial and Insurance Activities"),
        ("64929","Other credit granting n.e.c.","K – Financial and Insurance Activities"),
        ("64991","Security dealing on own account","K – Financial and Insurance Activities"),
        ("64992","Factoring","K – Financial and Insurance Activities"),
        ("64999","Other financial service activities n.e.c.","K – Financial and Insurance Activities"),
        ("65110","Life insurance","K – Financial and Insurance Activities"),
        ("65120","Non-life insurance","K – Financial and Insurance Activities"),
        ("65201","Life reinsurance","K – Financial and Insurance Activities"),
        ("65202","Non-life reinsurance","K – Financial and Insurance Activities"),
        ("65300","Pension funding","K – Financial and Insurance Activities"),
        ("66110","Administration of financial markets","K – Financial and Insurance Activities"),
        ("66120","Security and commodity contracts dealing activities","K – Financial and Insurance Activities"),
        ("66190","Other activities auxiliary to financial services","K – Financial and Insurance Activities"),
        ("66210","Risk and damage evaluation","K – Financial and Insurance Activities"),
        ("66220","Activities of insurance agents and brokers","K – Financial and Insurance Activities"),
        ("66290","Other activities auxiliary to insurance and pension funding","K – Financial and Insurance Activities"),
        ("66300","Fund management activities","K – Financial and Insurance Activities"),
        # Section L – Real Estate Activities
        ("68100","Buying and selling of own real estate","L – Real Estate Activities"),
        ("68201","Renting and operating of housing association real estate","L – Real Estate Activities"),
        ("68202","Letting and operating of conference and exhibition centres","L – Real Estate Activities"),
        ("68209","Other letting and operating of own or leased real estate","L – Real Estate Activities"),
        ("68310","Real estate agencies","L – Real Estate Activities"),
        ("68320","Management of real estate on a fee or contract basis","L – Real Estate Activities"),
        # Section M – Professional, Scientific and Technical Activities
        ("69101","Barristers at law","M – Professional, Scientific and Technical Activities"),
        ("69102","Solicitors","M – Professional, Scientific and Technical Activities"),
        ("69109","Activities of lawyers and solicitors n.e.c.","M – Professional, Scientific and Technical Activities"),
        ("69201","Accounting and auditing activities","M – Professional, Scientific and Technical Activities"),
        ("69202","Bookkeeping activities","M – Professional, Scientific and Technical Activities"),
        ("69203","Tax consultancy","M – Professional, Scientific and Technical Activities"),
        ("70100","Activities of head offices","M – Professional, Scientific and Technical Activities"),
        ("70210","Public relations and communications activities","M – Professional, Scientific and Technical Activities"),
        ("70221","Financial management","M – Professional, Scientific and Technical Activities"),
        ("70229","Management consultancy activities (other)","M – Professional, Scientific and Technical Activities"),
        ("71111","Architectural activities","M – Professional, Scientific and Technical Activities"),
        ("71112","Urban planning and landscape architectural activities","M – Professional, Scientific and Technical Activities"),
        ("71121","Engineering design activities for industrial process and production","M – Professional, Scientific and Technical Activities"),
        ("71122","Engineering related scientific and technical consulting activities","M – Professional, Scientific and Technical Activities"),
        ("71129","Other engineering activities","M – Professional, Scientific and Technical Activities"),
        ("71200","Technical testing and analysis","M – Professional, Scientific and Technical Activities"),
        ("72110","Research and experimental development on biotechnology","M – Professional, Scientific and Technical Activities"),
        ("72190","Other research and experimental development on natural sciences","M – Professional, Scientific and Technical Activities"),
        ("72200","Research and experimental development on social sciences and humanities","M – Professional, Scientific and Technical Activities"),
        ("73110","Advertising agencies","M – Professional, Scientific and Technical Activities"),
        ("73120","Media representation services","M – Professional, Scientific and Technical Activities"),
        ("73200","Market research and public opinion polling","M – Professional, Scientific and Technical Activities"),
        ("74100","Specialised design activities","M – Professional, Scientific and Technical Activities"),
        ("74201","Portrait photographic activities","M – Professional, Scientific and Technical Activities"),
        ("74202","Other specialist photography","M – Professional, Scientific and Technical Activities"),
        ("74203","Film processing","M – Professional, Scientific and Technical Activities"),
        ("74209","Photographic activities n.e.c.","M – Professional, Scientific and Technical Activities"),
        ("74300","Translation and interpretation activities","M – Professional, Scientific and Technical Activities"),
        ("74901","Patent and copyright activities","M – Professional, Scientific and Technical Activities"),
        ("74902","Quantity surveying activities","M – Professional, Scientific and Technical Activities"),
        ("74909","Other professional, scientific and technical activities n.e.c.","M – Professional, Scientific and Technical Activities"),
        ("75000","Veterinary activities","M – Professional, Scientific and Technical Activities"),
        # Section N – Administrative and Support Service Activities
        ("77110","Renting and leasing of cars and light motor vehicles","N – Administrative and Support Service Activities"),
        ("77120","Renting and leasing of trucks","N – Administrative and Support Service Activities"),
        ("77210","Renting and leasing of recreational and sports goods","N – Administrative and Support Service Activities"),
        ("77220","Renting of video tapes and disks","N – Administrative and Support Service Activities"),
        ("77230","Renting and leasing of other personal and household goods","N – Administrative and Support Service Activities"),
        ("77290","Renting and leasing of other tangible assets n.e.c.","N – Administrative and Support Service Activities"),
        ("77300","Renting and leasing of other machinery and equipment","N – Administrative and Support Service Activities"),
        ("77400","Leasing of intellectual property and similar products","N – Administrative and Support Service Activities"),
        ("78101","Motion picture, television and other theatrical casting activities","N – Administrative and Support Service Activities"),
        ("78109","Other activities of employment placement agencies","N – Administrative and Support Service Activities"),
        ("78200","Temporary employment agency activities","N – Administrative and Support Service Activities"),
        ("78300","Human resources provision and management of human resources functions","N – Administrative and Support Service Activities"),
        ("79110","Travel agency activities","N – Administrative and Support Service Activities"),
        ("79120","Tour operator activities","N – Administrative and Support Service Activities"),
        ("79901","Activities of tourist offices","N – Administrative and Support Service Activities"),
        ("79909","Other reservation service and related activities","N – Administrative and Support Service Activities"),
        ("80100","Private security activities","N – Administrative and Support Service Activities"),
        ("80200","Security systems service activities","N – Administrative and Support Service Activities"),
        ("80300","Investigation activities","N – Administrative and Support Service Activities"),
        ("81100","Combined facilities support activities","N – Administrative and Support Service Activities"),
        ("81210","General cleaning of buildings","N – Administrative and Support Service Activities"),
        ("81221","Window cleaning services","N – Administrative and Support Service Activities"),
        ("81222","Specialised cleaning services","N – Administrative and Support Service Activities"),
        ("81223","Furnace and chimney cleaning services","N – Administrative and Support Service Activities"),
        ("81229","Other building and industrial cleaning activities","N – Administrative and Support Service Activities"),
        ("81291","Disinfecting and exterminating activities","N – Administrative and Support Service Activities"),
        ("81299","Other cleaning services","N – Administrative and Support Service Activities"),
        ("81300","Landscape service activities","N – Administrative and Support Service Activities"),
        ("82110","Combined office administrative service activities","N – Administrative and Support Service Activities"),
        ("82190","Photocopying, document preparation and other specialised office support activities","N – Administrative and Support Service Activities"),
        ("82200","Activities of call centres","N – Administrative and Support Service Activities"),
        ("82300","Organisation of conventions and trade shows","N – Administrative and Support Service Activities"),
        ("82911","Activities of collection agencies","N – Administrative and Support Service Activities"),
        ("82912","Activities of credit bureaus","N – Administrative and Support Service Activities"),
        ("82920","Packaging activities","N – Administrative and Support Service Activities"),
        ("82990","Other business support service activities n.e.c.","N – Administrative and Support Service Activities"),
        # Section O – Public Administration
        ("84110","General public administration activities","O – Public Administration and Defence"),
        ("84120","Regulation of health care, education and other social services","O – Public Administration and Defence"),
        ("84130","Regulation of and contribution to more efficient operation of businesses","O – Public Administration and Defence"),
        ("84210","Foreign affairs","O – Public Administration and Defence"),
        ("84220","Defence activities","O – Public Administration and Defence"),
        ("84230","Justice and judicial activities","O – Public Administration and Defence"),
        ("84240","Public order and safety activities","O – Public Administration and Defence"),
        ("84250","Fire service activities","O – Public Administration and Defence"),
        ("84300","Compulsory social security activities","O – Public Administration and Defence"),
        # Section P – Education
        ("85100","Pre-primary education","P – Education"),
        ("85200","Primary education","P – Education"),
        ("85310","General secondary education","P – Education"),
        ("85320","Technical and vocational secondary education","P – Education"),
        ("85410","Post-secondary non-tertiary education","P – Education"),
        ("85421","First-degree level higher education","P – Education"),
        ("85422","Post-graduate level higher education","P – Education"),
        ("85510","Sports and recreation education","P – Education"),
        ("85520","Cultural education","P – Education"),
        ("85530","Driving school activities","P – Education"),
        ("85590","Other education n.e.c.","P – Education"),
        ("85600","Educational support activities","P – Education"),
        # Section Q – Human Health and Social Work
        ("86101","Hospital activities","Q – Human Health and Social Work"),
        ("86102","Medical nursing home activities","Q – Human Health and Social Work"),
        ("86210","General medical practice activities","Q – Human Health and Social Work"),
        ("86220","Specialist medical practice activities","Q – Human Health and Social Work"),
        ("86230","Dental practice activities","Q – Human Health and Social Work"),
        ("86900","Other human health activities","Q – Human Health and Social Work"),
        ("87100","Residential nursing care activities","Q – Human Health and Social Work"),
        ("87200","Residential care activities for learning difficulties","Q – Human Health and Social Work"),
        ("87300","Residential care activities for the elderly and disabled","Q – Human Health and Social Work"),
        ("87900","Other residential care activities n.e.c.","Q – Human Health and Social Work"),
        ("88100","Social work activities without accommodation for the elderly and disabled","Q – Human Health and Social Work"),
        ("88910","Child day-care activities","Q – Human Health and Social Work"),
        ("88990","Other social work activities without accommodation n.e.c.","Q – Human Health and Social Work"),
        # Section R – Arts, Entertainment and Recreation
        ("90010","Performing arts","R – Arts, Entertainment and Recreation"),
        ("90020","Support activities to performing arts","R – Arts, Entertainment and Recreation"),
        ("90030","Artistic creation","R – Arts, Entertainment and Recreation"),
        ("90040","Operation of arts facilities","R – Arts, Entertainment and Recreation"),
        ("91011","Library activities","R – Arts, Entertainment and Recreation"),
        ("91012","Archive activities","R – Arts, Entertainment and Recreation"),
        ("91020","Museum activities","R – Arts, Entertainment and Recreation"),
        ("91030","Operation of historical sites","R – Arts, Entertainment and Recreation"),
        ("91040","Botanical and zoological gardens and nature reserves activities","R – Arts, Entertainment and Recreation"),
        ("92000","Gambling and betting activities","R – Arts, Entertainment and Recreation"),
        ("93110","Operation of sports facilities","R – Arts, Entertainment and Recreation"),
        ("93120","Activities of sport clubs","R – Arts, Entertainment and Recreation"),
        ("93130","Fitness facilities","R – Arts, Entertainment and Recreation"),
        ("93191","Activities of racehorse owners","R – Arts, Entertainment and Recreation"),
        ("93199","Other sports activities","R – Arts, Entertainment and Recreation"),
        ("93210","Activities of amusement parks and theme parks","R – Arts, Entertainment and Recreation"),
        ("93290","Other amusement and recreation activities n.e.c.","R – Arts, Entertainment and Recreation"),
        # Section S – Other Service Activities
        ("94110","Activities of business and employers membership organisations","S – Other Service Activities"),
        ("94120","Activities of professional membership organisations","S – Other Service Activities"),
        ("94200","Activities of trade unions","S – Other Service Activities"),
        ("94910","Activities of religious organisations","S – Other Service Activities"),
        ("94920","Activities of political organisations","S – Other Service Activities"),
        ("94990","Activities of other membership organisations n.e.c.","S – Other Service Activities"),
        ("95110","Repair of computers and peripheral equipment","S – Other Service Activities"),
        ("95120","Repair of communication equipment","S – Other Service Activities"),
        ("95210","Repair of consumer electronics","S – Other Service Activities"),
        ("95220","Repair of household appliances","S – Other Service Activities"),
        ("95230","Repair of footwear and leather goods","S – Other Service Activities"),
        ("95240","Repair of furniture and home furnishings","S – Other Service Activities"),
        ("95250","Repair of watches, clocks and jewellery","S – Other Service Activities"),
        ("95290","Repair of personal and household goods n.e.c.","S – Other Service Activities"),
        ("96010","Washing and (dry-)cleaning of textile and fur products","S – Other Service Activities"),
        ("96020","Hairdressing and other beauty treatment","S – Other Service Activities"),
        ("96030","Funeral and related activities","S – Other Service Activities"),
        ("96040","Physical well-being activities","S – Other Service Activities"),
        ("96090","Other service activities n.e.c.","S – Other Service Activities"),
        # Section T – Activities of Households as Employers
        ("97000","Activities of households as employers of domestic personnel","T – Activities of Households as Employers"),
        ("98100","Undifferentiated goods-producing activities of private households","T – Activities of Households as Employers"),
        ("98200","Undifferentiated service-producing activities of private households","T – Activities of Households as Employers"),
        # Section U – Activities of Extraterritorial Organisations
        ("99000","Activities of extraterritorial organisations and bodies","U – Activities of Extraterritorial Organisations"),
    ]
    count = 0
    for sic_code, description, section in SIC_DATA:
        try:
            db_execute("""
                insert into public.company_type (sic_code, description, section)
                values (%s, %s, %s)
                on conflict (sic_code) do nothing
            """, (sic_code, description, section))
            count += 1
        except Exception:
            pass
    return count


# =========================================================
# API FUNCTIONS
# =========================================================
def api_get(path: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    session = get_http_session()
    limiter = get_rate_limiter()
    url = f"{BASE_URL}{path}"

    for attempt in range(1, 6):
        limiter.wait_if_needed()
        resp = session.get(url, params=params, timeout=30)

        if resp.status_code == 200:
            return resp.json()

        if resp.status_code in (429, 500, 502, 503, 504):
            retry_after = resp.headers.get("Retry-After")
            if retry_after:
                sleep_for = min(60, float(retry_after))
            else:
                sleep_for = min(60, 2 ** attempt)
            time.sleep(sleep_for)
            continue

        raise RuntimeError(f"API error {resp.status_code}: {resp.text}")

    raise RuntimeError("API request failed after retries.")


def search_companies_api(query_text: str) -> List[Dict[str, Any]]:
    payload = api_get("/search/companies", params={"q": query_text})
    return payload.get("items", [])[:SEARCH_RESULT_LIMIT]


def fetch_company_profile_api(company_number: str) -> Dict[str, Any]:
    return api_get(f"/company/{company_number}")


def fetch_officers_api(company_number: str) -> List[Dict[str, Any]]:
    payload = api_get(f"/company/{company_number}/officers")
    return payload.get("items", [])


# =========================================================
# REPOSITORY / CACHE LOGIC
# =========================================================
def upsert_search_query(query_text: str) -> str:
    normalized = normalize_query(query_text)

    row = db_fetchone("""
        insert into public.search_queries (query_text, normalized_query, last_searched_at)
        values (%s, %s, now())
        on conflict (normalized_query)
        do update set
            query_text = excluded.query_text,
            last_searched_at = now()
        returning id::text
    """, (query_text, normalized))

    return row[0]


def get_search_query_record(query_text: str) -> Optional[tuple]:
    normalized = normalize_query(query_text)
    return db_fetchone("""
        select id::text, query_text, normalized_query, last_searched_at
        from public.search_queries
        where normalized_query = %s
    """, (normalized,))


def get_cached_search_results(query_text: str) -> Tuple[Optional[str], List[tuple], Optional[datetime]]:
    record = get_search_query_record(query_text)
    if not record:
        return None, [], None

    query_id, _, _, last_searched_at = record

    rows = db_fetchall("""
        select
            c.company_number,
            c.title,
            c.company_status,
            c.company_type,
            c.date_of_creation,
            c.description,
            c.address_snippet,
            sr.rank_order
        from public.search_results sr
        join public.companies c on c.company_number = sr.company_number
        where sr.query_id = %s::uuid
        order by sr.rank_order asc
    """, (query_id,))

    return query_id, rows, last_searched_at


def upsert_company_from_search(item: Dict[str, Any]) -> None:
    db_execute("""
        insert into public.companies (
            company_number,
            title,
            company_status,
            company_type,
            date_of_creation,
            description,
            address_snippet,
            raw_search_json
        )
        values (%s, %s, %s, %s, %s, %s, %s, %s::jsonb)
        on conflict (company_number)
        do update set
            title = excluded.title,
            company_status = excluded.company_status,
            company_type = excluded.company_type,
            date_of_creation = excluded.date_of_creation,
            description = excluded.description,
            address_snippet = excluded.address_snippet,
            raw_search_json = excluded.raw_search_json
    """, (
        item.get("company_number"),
        item.get("title"),
        item.get("company_status"),
        item.get("company_type"),
        parse_date(item.get("date_of_creation")),
        item.get("description"),
        item.get("address_snippet"),
        to_json(item)
    ))


def replace_search_results(query_id: str, items: List[Dict[str, Any]]) -> None:
    db_execute("delete from public.search_results where query_id = %s::uuid", (query_id,))

    for idx, item in enumerate(items, start=1):
        upsert_company_from_search(item)
        db_execute("""
            insert into public.search_results (query_id, company_number, rank_order)
            values (%s::uuid, %s, %s)
            on conflict (query_id, company_number)
            do update set rank_order = excluded.rank_order
        """, (query_id, item.get("company_number"), idx))


def get_or_refresh_search_results(query_text: str) -> pd.DataFrame:
    query_id, cached_rows, last_searched_at = get_cached_search_results(query_text)

    if query_id and cached_rows and not is_stale(last_searched_at, CACHE_TTL_SEARCH_HOURS):
        return pd.DataFrame(cached_rows, columns=[
            "company_number", "title", "company_status", "company_type",
            "date_of_creation", "description", "address_snippet", "rank_order"
        ])

    if not query_id:
        query_id = upsert_search_query(query_text)
    else:
        db_execute("""
            update public.search_queries
            set last_searched_at = now()
            where id = %s::uuid
        """, (query_id,))

    items = search_companies_api(query_text)
    replace_search_results(query_id, items)

    query_id, cached_rows, _ = get_cached_search_results(query_text)

    return pd.DataFrame(cached_rows, columns=[
        "company_number", "title", "company_status", "company_type",
        "date_of_creation", "description", "address_snippet", "rank_order"
    ])


def get_company_record(company_number: str) -> Optional[tuple]:
    return db_fetchone("""
        select
            company_number,
            title,
            company_status,
            company_type,
            date_of_creation,
            description,
            address_snippet,
            registered_office_address,
            raw_profile_json,
            last_profile_fetched_at,
            region,
            sic_code,
            section_type
        from public.companies
        where company_number = %s
    """, (company_number,))


def upsert_company_profile(company_number: str, profile: Dict[str, Any]) -> None:
    registered_office_address = profile.get("registered_office_address") or {}

    address_parts = [
        registered_office_address.get("premises"),
        registered_office_address.get("address_line_1"),
        registered_office_address.get("address_line_2"),
        registered_office_address.get("locality"),
        registered_office_address.get("region"),
        registered_office_address.get("postal_code"),
        registered_office_address.get("country"),
    ]
    address_snippet = ", ".join([x for x in address_parts if x]) or None

    # Extract region from address
    region = registered_office_address.get("region") or None

    # Extract first SIC code and look up description + section
    sic_codes = profile.get("sic_codes") or []
    first_sic = sic_codes[0] if sic_codes else None
    sic_description, section_type = lookup_company_type(first_sic) if first_sic else (None, None)

    db_execute("""
        insert into public.companies (
            company_number,
            title,
            company_status,
            company_type,
            date_of_creation,
            description,
            address_snippet,
            registered_office_address,
            raw_profile_json,
            last_profile_fetched_at,
            region,
            sic_code,
            section_type
        )
        values (%s, %s, %s, %s, %s, %s, %s, %s::jsonb, %s::jsonb, now(), %s, %s, %s)
        on conflict (company_number)
        do update set
            title = excluded.title,
            company_status = excluded.company_status,
            company_type = excluded.company_type,
            date_of_creation = excluded.date_of_creation,
            description = excluded.description,
            address_snippet = excluded.address_snippet,
            registered_office_address = excluded.registered_office_address,
            raw_profile_json = excluded.raw_profile_json,
            last_profile_fetched_at = now(),
            region = excluded.region,
            sic_code = excluded.sic_code,
            section_type = excluded.section_type
    """, (
        company_number,
        profile.get("company_name") or profile.get("title"),
        profile.get("company_status"),
        profile.get("type"),
        parse_date(profile.get("date_of_creation")),
        profile.get("description"),
        address_snippet,
        to_json(registered_office_address),
        to_json(profile),
        region,
        first_sic,
        section_type,
    ))


def get_or_refresh_company_profile(company_number: str) -> Dict[str, Any]:
    record = get_company_record(company_number)
    if record:
        raw_profile_json = record[8]
        last_profile_fetched_at = record[9]
        if raw_profile_json and not is_stale(last_profile_fetched_at, CACHE_TTL_COMPANY_HOURS):
            return raw_profile_json

    profile = fetch_company_profile_api(company_number)
    upsert_company_profile(company_number, profile)
    return profile


def get_cached_officers(company_number: str) -> Tuple[List[tuple], Optional[datetime]]:
    rows = db_fetchall("""
        select
            officer_name,
            officer_role,
            appointed_on,
            resigned_on,
            nationality,
            occupation,
            country_of_residence,
            last_fetched_at
        from public.officers
        where company_number = %s
        order by appointed_on desc nulls last, officer_name asc
    """, (company_number,))

    if not rows:
        return [], None

    last_fetched_at = rows[0][7]
    return rows, last_fetched_at


def replace_officers(company_number: str, items: List[Dict[str, Any]]) -> None:
    db_execute("delete from public.officers where company_number = %s", (company_number,))

    for item in items:
        db_execute("""
            insert into public.officers (
                company_number,
                officer_name,
                officer_role,
                appointed_on,
                resigned_on,
                nationality,
                occupation,
                country_of_residence,
                raw_json,
                last_fetched_at
            )
            values (%s, %s, %s, %s, %s, %s, %s, %s, %s::jsonb, now())
        """, (
            company_number,
            item.get("name"),
            item.get("officer_role"),
            parse_date(item.get("appointed_on")),
            parse_date(item.get("resigned_on")),
            item.get("nationality"),
            item.get("occupation"),
            item.get("country_of_residence"),
            to_json(item)
        ))


def get_or_refresh_officers(company_number: str) -> pd.DataFrame:
    cached_rows, last_fetched_at = get_cached_officers(company_number)

    if cached_rows and not is_stale(last_fetched_at, CACHE_TTL_OFFICERS_HOURS):
        return pd.DataFrame(cached_rows, columns=[
            "officer_name", "officer_role", "appointed_on", "resigned_on",
            "nationality", "occupation", "country_of_residence", "last_fetched_at"
        ])

    items = fetch_officers_api(company_number)
    replace_officers(company_number, items)

    cached_rows, _ = get_cached_officers(company_number)
    return pd.DataFrame(cached_rows, columns=[
        "officer_name", "officer_role", "appointed_on", "resigned_on",
        "nationality", "occupation", "country_of_residence", "last_fetched_at"
    ])


# =========================================================
# BULK ADMIN LOGIC
# =========================================================
def ensure_bulk_job_tables():
    db_execute("""
        create table if not exists public.bulk_upload_jobs (
            id uuid primary key default gen_random_uuid(),
            uploaded_at timestamptz not null default now(),
            uploaded_file_name text,
            total_keywords integer not null default 0,
            processed_keywords integer not null default 0,
            success_count integer not null default 0,
            failed_count integer not null default 0,
            status text not null default 'running'
        )
    """)

    db_execute("""
        create table if not exists public.bulk_upload_job_items (
            id uuid primary key default gen_random_uuid(),
            job_id uuid not null references public.bulk_upload_jobs(id) on delete cascade,
            keyword text not null,
            normalized_keyword text not null,
            status text not null default 'pending',
            matched_company_number text,
            matched_title text,
            error_message text,
            created_at timestamptz not null default now()
        )
    """)

    db_execute("""
        create index if not exists idx_bulk_upload_job_items_job_id
        on public.bulk_upload_job_items(job_id)
    """)


def create_bulk_job(file_name: str, total_keywords: int) -> str:
    row = db_fetchone("""
        insert into public.bulk_upload_jobs (uploaded_file_name, total_keywords, status)
        values (%s, %s, 'running')
        returning id::text
    """, (file_name, total_keywords))
    return row[0]


def insert_bulk_job_items(job_id: str, keywords: List[str]):
    rows = [(job_id, kw, normalize_query(kw), "pending") for kw in keywords]
    conn = get_db_connection()
    with conn.cursor() as cur:
        cur.executemany("""
            insert into public.bulk_upload_job_items (job_id, keyword, normalized_keyword, status)
            values (%s::uuid, %s, %s, %s)
        """, rows)


def update_bulk_job_progress(job_id: str, processed_inc: int, success_inc: int, failed_inc: int):
    db_execute("""
        update public.bulk_upload_jobs
        set
            processed_keywords = processed_keywords + %s,
            success_count = success_count + %s,
            failed_count = failed_count + %s
        where id = %s::uuid
    """, (processed_inc, success_inc, failed_inc, job_id))


def finalize_bulk_job(job_id: str, status: str = "completed"):
    db_execute("""
        update public.bulk_upload_jobs
        set status = %s
        where id = %s::uuid
    """, (status, job_id))


def update_bulk_job_item(job_id: str, keyword: str, status: str, matched_company_number: Optional[str], matched_title: Optional[str], error_message: Optional[str]):
    db_execute("""
        update public.bulk_upload_job_items
        set
            status = %s,
            matched_company_number = %s,
            matched_title = %s,
            error_message = %s
        where job_id = %s::uuid
          and normalized_keyword = %s
    """, (status, matched_company_number, matched_title, error_message, job_id, normalize_query(keyword)))


def process_single_keyword(keyword: str) -> Dict[str, Any]:
    result = {
        "keyword": keyword,
        "status": "success",
        "matched_company_number": None,
        "matched_title": None,
        "error_message": None,
    }

    try:
        results_df = get_or_refresh_search_results(keyword)

        if results_df.empty:
            result["status"] = "no_result"
            return result

        first_row = results_df.iloc[0]
        company_number = first_row.get("company_number")
        title = first_row.get("title")

        result["matched_company_number"] = company_number
        result["matched_title"] = title

        # hydrate cache more deeply once we have the company number
        if company_number:
            get_or_refresh_company_profile(company_number)
            get_or_refresh_officers(company_number)

        return result

    except Exception as e:
        result["status"] = "failed"
        result["error_message"] = str(e)
        return result


def process_bulk_keywords(job_id: str, keywords: List[str], batch_size: int):
    total = len(keywords)
    progress = st.progress(0)
    status_placeholder = st.empty()
    result_preview_placeholder = st.empty()

    success_count = 0
    failed_count = 0
    processed = 0
    preview_rows = []

    for batch_index, batch in enumerate(chunk_list(keywords, batch_size), start=1):
        batch_success = 0
        batch_failed = 0

        for keyword in batch:
            outcome = process_single_keyword(keyword)
            processed += 1

            if outcome["status"] in ("success", "no_result"):
                batch_success += 1
                if outcome["status"] == "no_result":
                    preview_rows.append({
                        "keyword": keyword,
                        "status": "no_result",
                        "company_number": None,
                        "title": None,
                    })
                else:
                    preview_rows.append({
                        "keyword": keyword,
                        "status": "success",
                        "company_number": outcome["matched_company_number"],
                        "title": outcome["matched_title"],
                    })
            else:
                batch_failed += 1
                preview_rows.append({
                    "keyword": keyword,
                    "status": "failed",
                    "company_number": None,
                    "title": outcome["error_message"],
                })

            update_bulk_job_item(
                job_id=job_id,
                keyword=keyword,
                status=outcome["status"],
                matched_company_number=outcome["matched_company_number"],
                matched_title=outcome["matched_title"],
                error_message=outcome["error_message"],
            )

            progress.progress(min(processed / total, 1.0))
            status_placeholder.info(
                f"Processing batch {batch_index} • "
                f"{processed}/{total} keywords • "
                f"success/no-result: {success_count + batch_success} • "
                f"failed: {failed_count + batch_failed}"
            )

        success_count += batch_success
        failed_count += batch_failed
        update_bulk_job_progress(job_id, len(batch), batch_success, batch_failed)

        preview_df = pd.DataFrame(preview_rows[-20:])
        if not preview_df.empty:
            result_preview_placeholder.dataframe(preview_df, use_container_width=True, hide_index=True)

    finalize_bulk_job(job_id, "completed")
    return {
        "processed": processed,
        "success_count": success_count,
        "failed_count": failed_count,
        "preview_rows": preview_rows[-50:],
    }


# =========================================================
# SCORING
# =========================================================
def calculate_investment_score(profile: Dict[str, Any], officers_df: pd.DataFrame):
    score = 0
    reasons = []

    status = (profile.get("company_status") or "").lower()
    if status == "active":
        score += 25
        reasons.append("Company status is active (+25)")
    else:
        score -= 20
        reasons.append(f"Company status is {status or 'unknown'} (-20)")

    created = profile.get("date_of_creation")
    if created:
        try:
            created_dt = datetime.strptime(created, "%Y-%m-%d").date()
            age_years = (datetime.utcnow().date() - created_dt).days / 365.25
            if age_years >= 10:
                score += 20
                reasons.append("Company age is at least 10 years (+20)")
            elif age_years >= 5:
                score += 10
                reasons.append("Company age is at least 5 years (+10)")
            else:
                reasons.append("Company is relatively new (+0)")
        except Exception:
            reasons.append("Could not parse company age (+0)")

    active_officers = 0
    if not officers_df.empty:
        active_officers = officers_df["resigned_on"].isna().sum()

    if active_officers >= 3:
        score += 15
        reasons.append("Has 3 or more active officers (+15)")
    elif active_officers >= 1:
        score += 8
        reasons.append("Has at least 1 active officer (+8)")
    else:
        score -= 10
        reasons.append("No active officers found (-10)")

    company_type = (profile.get("type") or "").lower()
    if company_type in {"ltd", "plc", "llp"}:
        score += 5
        reasons.append("Common operating company type (+5)")

    return score, reasons


# =========================================================
# UI HELPERS
# =========================================================
def render_hero():
    st.markdown(
        """
        <div class="hero">
            <h1>UK Investment Companies Lookup</h1>
            <p>
                Search companies from GOV.UK Companies House, cache results into Supabase,
                and review company profile, officers, and a simple investment suitability score.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_sidebar():
    with st.sidebar:
        st.markdown("### App Settings")
        st.write(f"Search TTL: **{CACHE_TTL_SEARCH_HOURS}h**")
        st.write(f"Company TTL: **{CACHE_TTL_COMPANY_HOURS}h**")
        st.write(f"Officers TTL: **{CACHE_TTL_OFFICERS_HOURS}h**")
        st.write(f"Search result limit: **{SEARCH_RESULT_LIMIT}**")
        st.write(f"API cap used: **{API_MAX_REQUESTS_PER_5_MIN} / 5 min**")
        st.write(f"Admin batch size: **{ADMIN_BATCH_SIZE}**")
        st.markdown("---")
        st.caption(
            "Data is written when users search or when admin uploads CSV. "
            "If cache is fresh, the app reads from Supabase instead of hitting the API."
        )


def render_profile(profile: Dict[str, Any]):
    addr = profile.get("registered_office_address") or {}
    address_parts = [
        addr.get("premises"),
        addr.get("address_line_1"),
        addr.get("address_line_2"),
        addr.get("locality"),
        addr.get("region"),
        addr.get("postal_code"),
        addr.get("country"),
    ]
    address_text = ", ".join([x for x in address_parts if x]) or "-"

    # SIC code lookup
    sic_codes = profile.get("sic_codes") or []
    first_sic = sic_codes[0] if sic_codes else None
    sic_description, sic_section = lookup_company_type(first_sic) if first_sic else (None, None)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Company Number", profile.get("company_number", "-"))
    c2.metric("Status", profile.get("company_status", "-"))
    c3.metric("Type", profile.get("type", "-"))
    c4.metric("Region", addr.get("region") or "-")

    left, right = st.columns([1.15, 1])

    with left:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.markdown("#### Company Overview")
        st.write(f"**Name:** {profile.get('company_name', '-')}")
        st.write(f"**Date of creation:** {format_date(profile.get('date_of_creation'))}")
        st.write(f"**Jurisdiction:** {profile.get('jurisdiction', '-')}")
        st.write(f"**Company status:** {profile.get('company_status', '-')}")
        st.write(f"**Company type:** {profile.get('type', '-')}")
        st.write(f"**Has insolvency history:** {profile.get('has_insolvency_history', '-')}")
        st.write(f"**Registered office address:** {address_text}")
        if sic_codes:
            st.write(f"**SIC codes:** {', '.join(sic_codes)}")
        if sic_description:
            st.write(f"**SIC description:** {sic_description}")
        if sic_section:
            st.write(f"**Section:** {sic_section}")
        st.markdown("</div>", unsafe_allow_html=True)

    with right:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.markdown("#### Quick Tags")
        tags = [
            profile.get("company_status"),
            profile.get("type"),
            profile.get("jurisdiction"),
            addr.get("region"),
            sic_section,
        ]
        for tag in tags:
            if tag:
                st.markdown(f'<span class="label-pill">{tag}</span>', unsafe_allow_html=True)
        st.markdown("<br/>", unsafe_allow_html=True)
        st.caption("Tags are taken from the company profile response.")
        st.markdown("</div>", unsafe_allow_html=True)


def render_search_results(results_df: pd.DataFrame):
    st.markdown("### Search Results")

    show_df = results_df.copy()
    show_df["date_of_creation"] = show_df["date_of_creation"].astype(str)
    show_df = show_df.rename(columns={
        "company_number": "Company Number",
        "title": "Title",
        "company_status": "Status",
        "company_type": "Type",
        "date_of_creation": "Created",
        "address_snippet": "Address",
        "rank_order": "Rank"
    })

    st.dataframe(show_df, use_container_width=True, hide_index=True)


def build_company_options(results_df: pd.DataFrame) -> List[str]:
    options = []
    for _, row in results_df.iterrows():
        title = row.get("title") or "-"
        company_number = row.get("company_number") or "-"
        status = row.get("company_status") or "-"
        options.append(f"{title} ({company_number}) • {status}")
    return options


def extract_company_number(option_text: str) -> str:
    match = re.search(r"\(([A-Za-z0-9]+)\)", option_text)
    return match.group(1) if match else ""


def render_score(score: int, reasons: List[str]):
    st.markdown('<div class="score-box">', unsafe_allow_html=True)
    st.markdown(f"### {score_color(score)} Investment Suitability Score: {score}")
    st.caption("This is a simple rule-based score for a quick first pass, not formal financial advice.")
    for reason in reasons:
        st.write(f"- {reason}")
    st.markdown("</div>", unsafe_allow_html=True)


# =========================================================
# BROWSE PAGE (with Region + Section filters)
# =========================================================
def render_browse_page():
    render_hero()
    render_sidebar()

    st.markdown("## Browse Cached Companies")
    st.caption(
        "Browse all companies previously fetched and cached in the database. "
        "Filter by region and/or SIC section to narrow the results."
    )

    # Build filter options
    all_regions = get_all_regions()
    all_sections = get_all_sections()

    f1, f2, f3 = st.columns([1, 1, 1])
    with f1:
        selected_regions = st.multiselect("Filter by Region", all_regions, placeholder="All regions")
    with f2:
        selected_sections = st.multiselect("Filter by Section", all_sections, placeholder="All sections")
    with f3:
        selected_status = st.multiselect(
            "Filter by Status",
            ["active", "dissolved", "liquidation", "administration", "voluntary-arrangement", "converted-closed"],
            placeholder="All statuses"
        )

    # Build dynamic SQL
    conditions = []
    params: List[Any] = []

    if selected_regions:
        placeholders = ",".join(["%s"] * len(selected_regions))
        conditions.append(f"region in ({placeholders})")
        params.extend(selected_regions)

    if selected_sections:
        placeholders = ",".join(["%s"] * len(selected_sections))
        conditions.append(f"section_type in ({placeholders})")
        params.extend(selected_sections)

    if selected_status:
        placeholders = ",".join(["%s"] * len(selected_status))
        conditions.append(f"company_status in ({placeholders})")
        params.extend(selected_status)

    where_clause = "where " + " and ".join(conditions) if conditions else ""

    rows = db_fetchall(f"""
        select
            company_number,
            title,
            company_status,
            company_type,
            date_of_creation,
            address_snippet,
            region,
            sic_code,
            section_type
        from public.companies
        {where_clause}
        order by title asc
        limit 500
    """, tuple(params))

    if not rows:
        st.info("No companies found matching your filters. Try searching first to populate the database.")
        return

    df = pd.DataFrame(rows, columns=[
        "company_number", "title", "company_status", "company_type",
        "date_of_creation", "address_snippet", "region", "sic_code", "section_type"
    ])
    df["date_of_creation"] = df["date_of_creation"].astype(str)

    st.markdown(f"**{len(df)}** companies shown (max 500)")
    st.dataframe(
        df.rename(columns={
            "company_number": "Company #",
            "title": "Name",
            "company_status": "Status",
            "company_type": "Type",
            "date_of_creation": "Created",
            "address_snippet": "Address",
            "region": "Region",
            "sic_code": "SIC Code",
            "section_type": "Section",
        }),
        use_container_width=True,
        hide_index=True,
    )


# =========================================================
# MAIN USER PAGE
# =========================================================
def render_main_page():
    render_hero()
    render_sidebar()

    col1, col2 = st.columns([2.2, 1])
    with col1:
        query = st.text_input(
            "Search companies by keyword",
            placeholder="e.g. tesla, unilever, google, healthcare, fintech",
        )
    with col2:
        st.markdown("<div style='height: 28px;'></div>", unsafe_allow_html=True)
        run_search = st.button("Search Companies", use_container_width=True)

    if "last_query" not in st.session_state:
        st.session_state.last_query = ""
    if "results_df" not in st.session_state:
        st.session_state.results_df = pd.DataFrame()
    if "enriched_profiles" not in st.session_state:
        st.session_state.enriched_profiles = {}

    if run_search and query.strip():
        with st.spinner("Searching companies and checking cache..."):
            st.session_state.results_df = get_or_refresh_search_results(query.strip())
            st.session_state.last_query = query.strip()
            # Deep-fetch profiles to populate region/section data
            st.session_state.enriched_profiles = {}
            for _, row in st.session_state.results_df.iterrows():
                cn = row.get("company_number")
                if cn:
                    try:
                        p = get_or_refresh_company_profile(cn)
                        st.session_state.enriched_profiles[cn] = p
                    except Exception:
                        pass

    if st.session_state.last_query:
        st.markdown(
            f"<div class='small-muted'>Current query: <b>{st.session_state.last_query}</b></div>",
            unsafe_allow_html=True
        )

    results_df = st.session_state.results_df

    if results_df.empty:
        st.info("Search by keyword first. Data is written into Supabase when users search.")
        return

    # Build enriched dataframe with region + section from fetched profiles
    enriched = st.session_state.enriched_profiles
    if enriched:
        regions_map, sections_map = {}, {}
        for cn, p in enriched.items():
            addr = p.get("registered_office_address") or {}
            regions_map[cn] = addr.get("region")
            sic_codes = p.get("sic_codes") or []
            _, section = lookup_company_type(sic_codes[0]) if sic_codes else (None, None)
            sections_map[cn] = section
        results_df = results_df.copy()
        results_df["region"] = results_df["company_number"].map(regions_map)
        results_df["section"] = results_df["company_number"].map(sections_map)

        # Filter controls
        all_r = sorted([v for v in regions_map.values() if v])
        all_s = sorted(set([v for v in sections_map.values() if v]))
        fc1, fc2 = st.columns(2)
        sel_r = fc1.multiselect("Filter results by Region", all_r, key="sr_region")
        sel_s = fc2.multiselect("Filter results by Section", all_s, key="sr_section")

        if sel_r:
            results_df = results_df[results_df["region"].isin(sel_r)]
        if sel_s:
            results_df = results_df[results_df["section"].isin(sel_s)]

    render_search_results(results_df)

    options = build_company_options(results_df)
    selected = st.selectbox("Choose a company", options)

    selected_company_number = extract_company_number(selected)
    if not selected_company_number:
        st.warning("Could not read company number from selection.")
        return

    action_col1, action_col2, _ = st.columns([1, 1, 4])
    with action_col1:
        force_refresh_profile = st.button("Refresh Profile")
    with action_col2:
        force_refresh_officers = st.button("Refresh Officers")

    if force_refresh_profile:
        db_execute("""
            update public.companies
            set last_profile_fetched_at = null
            where company_number = %s
        """, (selected_company_number,))
        st.success("Company profile marked for refresh.")

    if force_refresh_officers:
        db_execute("""
            delete from public.officers
            where company_number = %s
        """, (selected_company_number,))
        st.success("Officers cache cleared.")

    with st.spinner("Loading company details..."):
        profile = get_or_refresh_company_profile(selected_company_number)

    with st.spinner("Loading officers..."):
        officers_df = get_or_refresh_officers(selected_company_number)

    score, reasons = calculate_investment_score(profile, officers_df)

    tab1, tab2, tab3 = st.tabs(["Overview", "Officers", "Raw JSON"])

    with tab1:
        render_profile(profile)
        render_score(score, reasons)

    with tab2:
        st.markdown("### Officers")
        if officers_df.empty:
            st.info("No officers found for this company.")
        else:
            show_officers = officers_df.drop(columns=["last_fetched_at"], errors="ignore").copy()
            show_officers = show_officers.rename(columns={
                "officer_name": "Officer Name",
                "officer_role": "Role",
                "appointed_on": "Appointed On",
                "resigned_on": "Resigned On",
                "nationality": "Nationality",
                "occupation": "Occupation",
                "country_of_residence": "Country of Residence",
            })
            st.dataframe(show_officers, use_container_width=True, hide_index=True)

            active_officers = 0
            if "resigned_on" in officers_df.columns:
                active_officers = officers_df["resigned_on"].isna().sum()

            m1, m2 = st.columns(2)
            m1.metric("Total Officers", len(officers_df))
            m2.metric("Active Officers", int(active_officers))

    with tab3:
        st.markdown("### Raw Company Profile JSON")
        st.json(profile)


# =========================================================
# HIDDEN ADMIN PAGE
# =========================================================
def render_admin_page():
    st.markdown(
        """
        <div class="admin-banner">
            <b>Hidden Admin Upload Page</b><br/>
            Access is controlled by URL token. This page is intentionally not shown in the main navigation.
        </div>
        """,
        unsafe_allow_html=True,
    )

    ensure_bulk_job_tables()

    # SIC Code Seeding
    st.markdown("## Seed SIC Code Lookup Table")
    st.caption(
        "Populate the `company_type` table with standard UK SIC 2007 codes. "
        "Safe to run multiple times – uses ON CONFLICT DO NOTHING."
    )
    if st.button("🌱 Seed SIC Codes (UK SIC 2007)", type="secondary"):
        with st.spinner("Seeding company_type table..."):
            count = seed_company_type_table()
        st.success(f"Done! Attempted to insert {count} SIC code rows (duplicates skipped automatically).")
    st.markdown("---")

    st.markdown("## Upload CSV Keywords")
    st.caption(
        "Upload a CSV containing company keywords. The app processes keywords in small batches, "
        "uses cache aggressively, and respects API rate limits."
    )

    uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

    col1, col2 = st.columns([1, 1])
    custom_batch_size = col1.number_input(
        "Batch size",
        min_value=10,
        max_value=500,
        value=ADMIN_BATCH_SIZE,
        step=10,
        help="Processing batch size inside one job. Keep it moderate for stability.",
    )
    max_rows = col2.number_input(
        "Max rows to process from file",
        min_value=1,
        max_value=500000,
        value=5000,
        step=100,
    )

    if uploaded_file is None:
        st.info("Upload a CSV with a keyword column.")
        return

    try:
        raw_bytes = uploaded_file.getvalue()
        text_data = raw_bytes.decode("utf-8-sig", errors="ignore")
        df = pd.read_csv(io.StringIO(text_data))
    except Exception as e:
        st.error(f"Could not read CSV: {e}")
        return

    if df.empty:
        st.warning("CSV is empty.")
        return

    keyword_col = detect_keyword_column(df)

    if not keyword_col:
        st.error(
            "Could not detect keyword column. Use one of these names: "
            "keyword, keywords, company, companies, company_name, query, name."
        )
        st.dataframe(df.head(10), use_container_width=True)
        return

    working_df = df[[keyword_col]].copy()
    working_df.columns = ["keyword"]
    working_df["keyword"] = working_df["keyword"].astype(str).str.strip()
    working_df = working_df[working_df["keyword"] != ""]
    working_df["normalized_keyword"] = working_df["keyword"].map(normalize_query)
    working_df = working_df.drop_duplicates(subset=["normalized_keyword"])
    working_df = working_df.head(int(max_rows))

    st.markdown("### Preview")
    st.write(f"Detected keyword column: **{keyword_col}**")
    st.write(f"Rows after cleaning and dedup: **{len(working_df)}**")
    st.dataframe(working_df[["keyword"]].head(20), use_container_width=True, hide_index=True)

    if working_df.empty:
        st.warning("No usable keyword rows after cleaning.")
        return

    if st.button("Start Bulk Processing", type="primary", use_container_width=True):
        keywords = working_df["keyword"].tolist()
        job_id = create_bulk_job(uploaded_file.name, len(keywords))
        insert_bulk_job_items(job_id, keywords)

        st.success(f"Bulk job created: {job_id}")

        with st.spinner("Processing bulk job..."):
            result = process_bulk_keywords(job_id, keywords, int(custom_batch_size))

        st.success("Bulk processing completed.")
        st.write(f"Processed: **{result['processed']}**")
        st.write(f"Success / no-result: **{result['success_count']}**")
        st.write(f"Failed: **{result['failed_count']}**")

        preview_df = pd.DataFrame(result["preview_rows"])
        if not preview_df.empty:
            st.markdown("### Recent Results")
            st.dataframe(preview_df, use_container_width=True, hide_index=True)

    st.markdown("### Recent Upload Jobs")
    jobs = db_fetchall("""
        select
            id::text,
            uploaded_at,
            uploaded_file_name,
            total_keywords,
            processed_keywords,
            success_count,
            failed_count,
            status
        from public.bulk_upload_jobs
        order by uploaded_at desc
        limit 20
    """)
    if jobs:
        jobs_df = pd.DataFrame(jobs, columns=[
            "job_id",
            "uploaded_at",
            "file_name",
            "total_keywords",
            "processed_keywords",
            "success_count",
            "failed_count",
            "status",
        ])
        st.dataframe(jobs_df, use_container_width=True, hide_index=True)
    else:
        st.info("No bulk jobs yet.")


# =========================================================
# ROUTING
# =========================================================
def main():
    if not COMPANIES_HOUSE_API_KEY or not SUPABASE_DB_URL or not ADMIN_ACCESS_TOKEN:
        show_config_error()
        return

    # Ensure company_type table and extra columns exist on every startup
    ensure_company_type_table()

    params = st.query_params
    page = str(params.get("page", "home")).lower()
    token = str(params.get("token", ""))

    if page == "admin":
        if token != ADMIN_ACCESS_TOKEN:
            st.error("Invalid or missing admin token.")
            st.stop()
        render_admin_page()
        return

    if page == "browse":
        render_browse_page()
        return

    # Navigation tabs at the top of main page
    nav_col1, nav_col2, _ = st.columns([1, 1, 4])
    with nav_col1:
        if st.button("🔍 Search", use_container_width=True):
            st.query_params.update({"page": "home"})
            st.rerun()
    with nav_col2:
        if st.button("📋 Browse Database", use_container_width=True):
            st.query_params.update({"page": "browse"})
            st.rerun()

    render_main_page()


main()