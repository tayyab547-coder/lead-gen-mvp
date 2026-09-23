import os
import re
import json
import time
import requests
import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from bs4 import BeautifulSoup
from groq import Groq

try:
    from googlesearch import search as google_search
    GOOGLE_OK = True
except ImportError:
    GOOGLE_OK = False

try:
    from ddgs import DDGS
    DDGS_OK = True
except ImportError:
    DDGS_OK = False

try:
    GROQ_API_KEY = st.secrets["GROQ_API_KEY"]
except (FileNotFoundError, KeyError):
    load_dotenv()
    GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    st.error("GROQ_API_KEY not found. Check your .env file or Streamlit Secrets.")
    st.stop()

client = Groq(api_key=GROQ_API_KEY)

COLUMNS = [
    "Business Name", "Business Category", "Phone Number", "Website",
    "Street Address", "City", "State", "ZIP Code", "Google Maps URL",
    "Rating", "Review Count", "Email", "Facebook", "Instagram",
    "LinkedIn", "Source URL",
]

GROQ_MODELS = [
    "openai/gpt-oss-120b",
    "openai/gpt-oss-20b",
    "qwen/qwen3.6-27b",
]

def google_web_search(query: str, max_results: int = 25):
    results = []
    try:
        for url in google_search(query, num_results=max_results, lang="en", region="us", sleep_interval=2, advanced=False):
            results.append({"title": "", "href": url, "body": ""})
    except Exception as e:
        st.warning(f"Google search error: {e}")
    return results

def ddg_web_search(query: str, max_results: int = 25):
    results = []
    try:
        with DDGS() as ddgs:
            for r in ddgs.text(query, max_results=max_results):
                results.append({"title": r.get("title", ""), "href": r.get("href", ""), "body": r.get("body", "")})
    except Exception:
        pass
    return results

def web_search(query: str, max_results: int = 25):
    if GOOGLE_OK:
        results = google_web_search(query, max_results)
        if results:
            return results
    if DDGS_OK:
        return ddg_web_search(query, max_results)
    return []

def multi_search(industry: str, location: str, target: int):
    queries = [
        f"{industry} in {location}",
        f"best {industry} {location}",
        f"top {industry} companies {location} contact",
        f"{industry} {location} phone number address",
        f"{industry} near {location} website",
    ]
    all_results = []
    seen_urls = set()
    for q in queries:
        if len(all_results) >= target * 3:
            break
        for r in web_search(q, max_results=25):
            if r["href"] and r["href"] not in seen_urls:
                seen_urls.add(r["href"])
                all_results.append(r)
        time.sleep(1)
    return all_results

PROMPT = """You are a lead data extractor. Return ONLY a JSON object.

From the search results below, extract business leads matching the industry and location.

Industry: {industry}
Location: {location}

RULES:
1. Extract EVERY distinct business you can identify.
2. Only use information that actually exists in the results. NEVER invent data.
3. If a field is not found, use an empty string "".
4. "Source URL" must be the URL where the info was found.
5. Return UP TO {limit} businesses.

Return a JSON object with a single key "leads" whose value is an array of business objects.
Each business object must have these exact keys:
"Business Name", "Business Category", "Phone Number", "Website",
"Street Address", "City", "State", "ZIP Code", "Google Maps URL",
"Rating", "Review Count", "Email", "Facebook", "Instagram",
"LinkedIn", "Source URL"

Search results:
{results}
"""

def call_groq(prompt: str, json_mode: bool = False):
    last_error = None
    for model_name in GROQ_MODELS:
        try:
            kwargs = {
                "model": model_name,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.1,
            }
            if json_mode:
                kwargs["response_format"] = {"type": "json_object"}
            response = client.chat.completions.create(**kwargs)
            return response.choices[0].message.content.strip()
        except Exception as e:
            last_error = f"{model_name}: {e}"
            continue
    raise RuntimeError(f"All Groq models failed. Last error -> {last_error}")

def extract_leads(industry, location, results, limit):
    if not results:
        return []
    trimmed = results[:20]
    results_text = "\n\n".join(
        f"Title: {r['title']}\nURL: {r['href']}\nSnippet: {r['body']}"
        for r in trimmed
    )
    prompt = PROMPT.format(industry=industry, location=location, limit=limit, results=results_text)
    for json_mode in (True, False):
        try:
            text = call_groq(prompt, json_mode=json_mode)
            text = re.sub(r"^```(json)?", "", text).strip()
            text = re.sub(r"```$", "", text).strip()
            parsed = json.loads(text)
            if isinstance(parsed, dict):
                data = parsed.get("leads") or parsed.get("Leads") or []
            else:
                data = parsed
            if data:
                return data
        except Exception as e:
            st.warning(f"Extraction attempt (json_mode={json_mode}) failed: {e}")
            continue
    st.error("Could not extract any leads. Try again.")
    return []

def fetch_website_text(url: str, max_chars: int = 3000) -> str:
    if not url or not url.startswith("http"):
        return ""
    try:
        headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
        resp = requests.get(url, headers=headers, timeout=8)
        if resp.status_code != 200:
            return ""
        soup = BeautifulSoup(resp.text, "html.parser")
        for tag in soup(["script", "style", "noscript"]):
            tag.decompose()
        text = soup.get_text(separator=" ", strip=True)
        return text[:max_chars]
    except Exception:
        return ""

ENRICH_PROMPT = """You are extracting contact details from a business website's homepage.

Business Name: {name}
Website: {website}

Here is the visible text of the page:
-----
{page_text}
-----

Extract the following fields. Only use info that is actually on the page.
If a field is not present, use "".

Return ONLY a JSON object (no markdown) with these exact keys:
"Phone Number", "Email", "Street Address", "City", "State", "ZIP Code",
"Facebook", "Instagram", "LinkedIn", "Google Maps URL"
"""

def enrich_lead(lead: dict) -> dict:
    website = lead.get("Website", "")
    if not website:
        return lead
    page_text = fetch_website_text(website)
    if not page_text:
        return lead
    prompt = ENRICH_PROMPT.format(name=lead.get("Business Name", ""), website=website, page_text=page_text)
    try:
        text = call_groq(prompt)
        text = re.sub(r"^```(json)?", "", text).strip()
        text = re.sub(r"```$", "", text).strip()
        extra = json.loads(text)
        for k, v in extra.items():
            if k in lead and not lead.get(k) and v:
                lead[k] = v
    except Exception:
        pass
    return lead

def normalize_phone(p):
    if not p:
        return ""
    digits = re.sub(r"\D", "", str(p))
    if len(digits) == 10:
        return f"({digits[:3]}) {digits[3:6]}-{digits[6:]}"
    if len(digits) == 11 and digits.startswith("1"):
        return f"({digits[1:4]}) {digits[4:7]}-{digits[7:]}"
    return str(p).strip()

def clean_df(rows):
    df = pd.DataFrame(rows, columns=COLUMNS)
    for c in df.columns:
        df[c] = df[c].astype(str).str.strip().replace({"nan": "", "None": ""})
    df["Phone Number"] = df["Phone Number"].apply(normalize_phone)
    df = df[(df["Business Name"] != "") | (df["Website"] != "")]
    df = df.drop_duplicates(subset=["Business Name", "Website"], keep="first")
    df = df.reset_index(drop=True)
    return df

st.set_page_config(page_title="Lead Generation Assistant", layout="wide")
st.title("Lead Generation Assistant")
st.caption("MVP — Google search (free scrape), Groq AI extraction, website enrichment.")

col1, col2, col3 = st.columns([2, 2, 1])
with col1:
    industry = st.text_input("Industry", value="Med Spas")
with col2:
    location = st.text_input("Location", value="California")
with col3:
    num_leads = st.number_input("Number of Leads", min_value=1, max_value=50, value=10)

if st.button("Find Leads", type="primary"):
    if not industry or not location:
        st.warning("Please fill in Industry and Location.")
        st.stop()

    with st.spinner("Searching the web (Google + DuckDuckGo)..."):
        results = multi_search(industry, location, target=num_leads)

    st.info(f"Collected {len(results)} unique pages to analyze.")

    if not results:
        st.error("No search results found. Try again in a minute.")
        st.stop()

    with st.spinner(f"Extracting leads with AI ({len(results)} pages)..."):
        rows = extract_leads(industry, location, results, num_leads)

    if not rows:
        st.error("Could not extract any leads. Try again.")
        st.stop()

    df = clean_df(rows)

    progress = st.progress(0, text="Enriching leads from their websites...")
    enriched_rows = df.to_dict(orient="records")
    for i, lead in enumerate(enriched_rows):
        if lead.get("Website"):
            enriched_rows[i] = enrich_lead(lead)
        progress.progress((i + 1) / len(enriched_rows),
                          text=f"Enriched {i+1}/{len(enriched_rows)}: {lead.get('Business Name','')[:40]}")
    progress.empty()

    df = clean_df(enriched_rows)
    st.session_state["df"] = df

if "df" in st.session_state:
    df = st.session_state["df"]
    st.success(f"Found {len(df)} businesses")
    st.dataframe(df, use_container_width=True)

    excel_bytes = None
    try:
        from io import BytesIO
        buf = BytesIO()
        with pd.ExcelWriter(buf, engine="openpyxl") as writer:
            df.to_excel(writer, index=False, sheet_name="Leads")
        excel_bytes = buf.getvalue()
    except Exception as e:
        st.warning(f"Excel export error: {e}")

    if excel_bytes:
        st.download_button(
            label="Download Excel",
            data=excel_bytes,
            file_name="leads.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )
