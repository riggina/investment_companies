import os
import re
import json
import time
from datetime import datetime, timezone, timedelta
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import requests
import streamlit as st
from dotenv import load_dotenv
import psycopg


load_dotenv()

COMPANIES_HOUSE_API_KEY = os.getenv("COMPANIES_HOUSE_API_KEY", "").strip()
SUPABASE_DB_URL = os.getenv("SUPABASE_DB_URL", "").strip()

CACHE_TTL_SEARCH_HOURS = int(os.getenv("CACHE_TTL_SEARCH_HOURS", "24"))
CACHE_TTL_COMPANY_HOURS = int(os.getenv("CACHE_TTL_COMPANY_HOURS", "168"))
CACHE_TTL_OFFICERS_HOURS = int(os.getenv("CACHE_TTL_OFFICERS_HOURS", "168"))
SEARCH_RESULT_LIMIT = int(os.getenv("SEARCH_RESULT_LIMIT", "10"))

BASE_URL = "https://api.company-information.service.gov.uk"

if not COMPANIES_HOUSE_API_KEY:
    raise RuntimeError("Missing COMPANIES_HOUSE_API_KEY in environment.")
if not SUPABASE_DB_URL:
    raise RuntimeError("Missing SUPABASE_DB_URL in environment.")


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
        "User-Agent": "Streamlit Companies House Lookup"
    })
    return session


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


def search_companies(query_text: str) -> List[Dict[str, Any]]:
    payload = api_get("/search/companies", params={"q": query_text})
    return payload.get("items", [])[:SEARCH_RESULT_LIMIT]


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

    items = search_companies(query_text)
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


def fetch_company_profile(company_number: str) -> Dict[str, Any]:
    return api_get(f"/company/{company_number}")


def get_or_refresh_company_profile(company_number: str) -> Dict[str, Any]:
    record = get_company_record(company_number)
    if record:
        raw_profile_json = record[8]
        last_profile_fetched_at = record[9]
        if raw_profile_json and not is_stale(last_profile_fetched_at, CACHE_TTL_COMPANY_HOURS):
            return raw_profile_json

    profile = fetch_company_profile(company_number)
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


def fetch_officers(company_number: str) -> List[Dict[str, Any]]:
    payload = api_get(f"/company/{company_number}/officers")
    return payload.get("items", [])


def get_or_refresh_officers(company_number: str) -> pd.DataFrame:
    cached_rows, last_fetched_at = get_cached_officers(company_number)

    if cached_rows and not is_stale(last_fetched_at, CACHE_TTL_OFFICERS_HOURS):
        return pd.DataFrame(cached_rows, columns=[
            "officer_name", "officer_role", "appointed_on", "resigned_on",
            "nationality", "occupation", "country_of_residence", "last_fetched_at"
        ])

    items = fetch_officers(company_number)
    replace_officers(company_number, items)

    cached_rows, _ = get_cached_officers(company_number)
    return pd.DataFrame(cached_rows, columns=[
        "officer_name", "officer_role", "appointed_on", "resigned_on",
        "nationality", "occupation", "country_of_residence", "last_fetched_at"
    ])


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


def render_profile(profile: Dict[str, Any]) -> None:
    st.subheader("Company Profile")

    col1, col2, col3 = st.columns(3)
    col1.metric("Company Number", profile.get("company_number", "-"))
    col2.metric("Status", profile.get("company_status", "-"))
    col3.metric("Type", profile.get("type", "-"))

    st.write(f"**Name:** {profile.get('company_name', '-')}")
    st.write(f"**Date of creation:** {profile.get('date_of_creation', '-')}")
    st.write(f"**Jurisdiction:** {profile.get('jurisdiction', '-')}")
    st.write(f"**Can file:** {profile.get('can_file', '-')}")
    st.write(f"**Has insolvency history:** {profile.get('has_insolvency_history', '-')}")

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
    address_text = ", ".join([x for x in address_parts if x])
    st.write(f"**Registered office address:** {address_text or '-'}")


def main():
    st.set_page_config(page_title="UK Companies Lookup", layout="wide")
    st.title("UK Companies Lookup")
    st.caption("Search UK company data, cache it in Supabase, and avoid duplicate API hits.")

    with st.sidebar:
        st.header("Settings")
        st.write(f"Search TTL: {CACHE_TTL_SEARCH_HOURS} hours")
        st.write(f"Company TTL: {CACHE_TTL_COMPANY_HOURS} hours")
        st.write(f"Officers TTL: {CACHE_TTL_OFFICERS_HOURS} hours")
        st.write(f"Max search rows: {SEARCH_RESULT_LIMIT}")

    query = st.text_input("Search companies by keyword", placeholder="e.g. tesla, unilever, google")

    if not query:
        st.info("Type a company keyword to search.")
        return

    with st.spinner("Searching companies..."):
        results_df = get_or_refresh_search_results(query)

    if results_df.empty:
        st.warning("No companies found.")
        return

    st.subheader("Search Results")
    show_df = results_df.copy().rename(columns={
        "company_number": "Company Number",
        "title": "Title",
        "company_status": "Status",
        "company_type": "Type",
        "date_of_creation": "Created",
        "address_snippet": "Address",
        "rank_order": "Rank"
    })
    st.dataframe(show_df, use_container_width=True)

    company_options = [
        f"{row['title']} ({row['company_number']})"
        for _, row in results_df.iterrows()
    ]
    selected = st.selectbox("Choose a company", company_options)
    selected_company_number = selected.split("(")[-1].replace(")", "").strip()

    col_a, col_b = st.columns([1, 1])
    refresh_profile = col_a.button("Force refresh company profile")
    refresh_officers = col_b.button("Force refresh officers")

    if refresh_profile:
        db_execute("""
            update public.companies
            set last_profile_fetched_at = null
            where company_number = %s
        """, (selected_company_number,))

    if refresh_officers:
        db_execute("""
            delete from public.officers
            where company_number = %s
        """, (selected_company_number,))

    with st.spinner("Loading company profile..."):
        profile = get_or_refresh_company_profile(selected_company_number)

    with st.spinner("Loading officers..."):
        officers_df = get_or_refresh_officers(selected_company_number)

    render_profile(profile)

    score, reasons = calculate_investment_score(profile, officers_df)

    st.subheader("Investment Suitability")
    st.metric("Score", score)
    for reason in reasons:
        st.write(f"- {reason}")

    st.subheader("Officers")
    if officers_df.empty:
        st.info("No officers found.")
    else:
        st.dataframe(
            officers_df.drop(columns=["last_fetched_at"], errors="ignore"),
            use_container_width=True
        )

    with st.expander("Raw company profile JSON"):
        st.json(profile)


if __name__ == "__main__":
    main()