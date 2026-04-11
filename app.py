import json
import re
import time
from datetime import datetime, timezone, timedelta
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import psycopg
import requests
import streamlit as st


# =========================================================
# CONFIG
# =========================================================
def get_secret(key: str, default: Optional[str] = None) -> Optional[str]:
    if key in st.secrets:
        return st.secrets[key]
    return default


COMPANIES_HOUSE_API_KEY = (get_secret("COMPANIES_HOUSE_API_KEY", "") or "").strip()
SUPABASE_DB_URL = (get_secret("SUPABASE_DB_URL", "") or "").strip()

CACHE_TTL_SEARCH_HOURS = int(get_secret("CACHE_TTL_SEARCH_HOURS", "24"))
CACHE_TTL_COMPANY_HOURS = int(get_secret("CACHE_TTL_COMPANY_HOURS", "168"))
CACHE_TTL_OFFICERS_HOURS = int(get_secret("CACHE_TTL_OFFICERS_HOURS", "168"))
SEARCH_RESULT_LIMIT = int(get_secret("SEARCH_RESULT_LIMIT", "10"))

BASE_URL = "https://api.company-information.service.gov.uk"

if not COMPANIES_HOUSE_API_KEY:
    raise RuntimeError("Missing COMPANIES_HOUSE_API_KEY in Streamlit secrets.")
if not SUPABASE_DB_URL:
    raise RuntimeError("Missing SUPABASE_DB_URL in Streamlit secrets.")


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
        max-width: 1350px;
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
        padding: 1rem 1rem 0.75rem 1rem;
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


def safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except Exception:
        return default


def score_color(score: int) -> str:
    if score >= 45:
        return "🟢"
    if score >= 20:
        return "🟡"
    return "🔴"


# =========================================================
# CONNECTIONS
# =========================================================
@st.cache_resource
def get_db_connection():
    conn = psycopg.connect(SUPABASE_DB_URL, autocommit=True)
    return conn


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
# API FUNCTIONS
# =========================================================
def api_get(path: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    session = get_http_session()
    url = f"{BASE_URL}{path}"

    for attempt in range(1, 6):
        resp = session.get(url, params=params, timeout=30)

        if resp.status_code == 200:
            return resp.json()

        if resp.status_code in (429, 500, 502, 503, 504):
            sleep_for = min(30, 2 ** attempt)
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
            last_profile_fetched_at
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
            last_profile_fetched_at
        )
        values (%s, %s, %s, %s, %s, %s, %s, %s::jsonb, %s::jsonb, now())
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
            last_profile_fetched_at = now()
    """, (
        company_number,
        profile.get("company_name") or profile.get("title"),
        profile.get("company_status"),
        profile.get("type"),
        parse_date(profile.get("date_of_creation")),
        profile.get("description"),
        address_snippet,
        to_json(registered_office_address),
        to_json(profile)
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
# SCORING
# =========================================================
def calculate_investment_score(profile: Dict[str, Any], officers_df: pd.DataFrame) -> Tuple[int, List[str]]:
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
    else:
        reasons.append("Creation date unavailable (+0)")

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
        st.markdown("---")
        st.markdown("### Data Flow")
        st.caption(
            "Data is stored when users search. If cached data exists and is still fresh, "
            "the app reads from Supabase instead of hitting Companies House again."
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

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Company Number", profile.get("company_number", "-"))
    c2.metric("Status", profile.get("company_status", "-"))
    c3.metric("Type", profile.get("type", "-"))
    c4.metric("Can File", str(profile.get("can_file", "-")))

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
        st.markdown("</div>", unsafe_allow_html=True)

    with right:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.markdown("#### Quick Tags")
        tags = [
            profile.get("company_status"),
            profile.get("type"),
            profile.get("jurisdiction"),
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
# MAIN
# =========================================================
def main():
    render_hero()
    render_sidebar()

    col1, col2 = st.columns([2.2, 1])
    with col1:
        query = st.text_input(
            "Search companies by keyword",
            placeholder="e.g. tesla, unilever, google, healthcare, fintech",
            label_visibility="visible",
        )
    with col2:
        st.markdown("<div style='height: 28px;'></div>", unsafe_allow_html=True)
        run_search = st.button("Search Companies", use_container_width=True)

    if "last_query" not in st.session_state:
        st.session_state.last_query = ""
    if "results_df" not in st.session_state:
        st.session_state.results_df = pd.DataFrame()

    if run_search and query.strip():
        with st.spinner("Searching companies and checking cache..."):
            st.session_state.results_df = get_or_refresh_search_results(query.strip())
            st.session_state.last_query = query.strip()

    if st.session_state.last_query:
        st.markdown(
            f"<div class='small-muted'>Current query: <b>{st.session_state.last_query}</b></div>",
            unsafe_allow_html=True
        )

    results_df = st.session_state.results_df

    if results_df.empty:
        st.info("Search by keyword first. Data is written into Supabase when users search.")
        return

    render_search_results(results_df)

    options = build_company_options(results_df)
    selected = st.selectbox("Choose a company", options)

    selected_company_number = extract_company_number(selected)
    if not selected_company_number:
        st.warning("Could not read company number from selection.")
        return

    action_col1, action_col2, action_col3 = st.columns([1, 1, 4])
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


main()