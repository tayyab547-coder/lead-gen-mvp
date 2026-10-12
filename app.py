import os
import re
import json
import time
import uuid
import base64
import concurrent.futures
import requests
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
from datetime import datetime
from io import BytesIO
from urllib.parse import urlparse, urljoin, quote_plus
from dotenv import load_dotenv
from bs4 import BeautifulSoup
from groq import Groq

from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak
)

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

load_dotenv()

try:
    GROQ_API_KEY = st.secrets["GROQ_API_KEY"]
except (FileNotFoundError, KeyError):
    GROQ_API_KEY = os.getenv("GROQ_API_KEY")

try:
    GCP_JSON_B64 = st.secrets["GCP_JSON_B64"]
except (FileNotFoundError, KeyError):
    GCP_JSON_B64 = os.getenv("GCP_JSON_B64", "")

if not GROQ_API_KEY:
    st.error("GROQ_API_KEY not found. Check your .env or Streamlit Secrets.")
    st.stop()

client = Groq(api_key=GROQ_API_KEY)

HEADERS = {
    "User-Agent": "LeadGenApp/1.0 (educational)",
    "Accept-Language": "en-US,en;q=0.9",
}

SEARCH_CACHE = {}
GOOGLE_BLOCKED_UNTIL = 0
LAST_CALL_TIME = 0

HISTORY_DIR = "history"
os.makedirs(HISTORY_DIR, exist_ok=True)

COLUMNS = [
    "Business Name", "Business Category", "Lead Score", "AI Summary",
    "Phone Number", "Email", "Website",
    "Street Address", "City", "State", "ZIP Code", "Country",
    "Google Maps URL", "Rating", "Review Count",
    "Facebook Page", "Instagram", "LinkedIn Company URL", "YouTube", "Twitter",
    "Executive Name", "Executive Title", "Executive LinkedIn",
    "Executive Email", "Email Pattern", "Executive Source",
    "Source URL",
    "Audit Score", "Audit Grade", "Issues", "Pitch Angle",
    "Outreach Status", "Duplicate", "First Seen"
]

GROQ_MODELS = [
    "llama-3.3-70b-versatile",
    "llama-3.1-8b-instant",
    "openai/gpt-oss-120b",
]

EMAIL_TEMPLATES = {
    "Cold Intro": "Short personalized cold outreach. Under 120 words. Soft CTA for 15-min call.",
    "Follow-up": "Polite follow-up. Under 100 words.",
    "LinkedIn Message": "LinkedIn connection under 300 chars.",
    "Partnership Pitch": "Partnership pitch. Under 150 words.",
    "Re-engagement": "Re-engage past contact. Under 100 words.",
}

TONES = ["Professional", "Friendly", "Casual", "Direct"]
OUTREACH_OPTIONS = ["New", "Contacted", "Replied", "Meeting Booked", "Not Interested"]

# ============================================================
# GOOGLE SHEETS
# ============================================================
SHEET_NAME = "Lead Gen Results"


def get_creds_dict():
    if not GCP_JSON_B64:
        return None
    try:
        raw = base64.b64decode(GCP_JSON_B64).decode("utf-8")
        return json.loads(raw)
    except Exception:
        try:
            return json.loads(GCP_JSON_B64)
        except Exception:
            return None


def get_gsheet():
    try:
        import gspread
        from oauth2client.service_account import ServiceAccountCredentials
        creds_dict = get_creds_dict()
        if not creds_dict:
            return None, "No credentials"
        scope = ["https://spreadsheets.google.com/feeds",
                 "https://www.googleapis.com/auth/drive"]
        c = ServiceAccountCredentials.from_json_keyfile_dict(creds_dict, scope)
        gc = gspread.authorize(c)
        try:
            sh = gc.open(SHEET_NAME)
        except gspread.SpreadsheetNotFound:
            sh = gc.create(SHEET_NAME)
        return sh.sheet1, None
    except Exception as e:
        return None, str(e)[:100]


def save_to_sheet(df, tab_name=None):
    """
    Save DataFrame to a NEW tab in the sheet.
    If tab_name is given, uses it. Otherwise creates a timestamped name.
    """
    try:
        import gspread
        from oauth2client.service_account import ServiceAccountCredentials
        creds_dict = get_creds_dict()
        if not creds_dict:
            return False, "No credentials"

        scope = ["https://spreadsheets.google.com/feeds",
                 "https://www.googleapis.com/auth/drive"]
        c = ServiceAccountCredentials.from_json_keyfile_dict(creds_dict, scope)
        gc = gspread.authorize(c)

        try:
            sh = gc.open(SHEET_NAME)
        except gspread.SpreadsheetNotFound:
            return False, f"Sheet '{SHEET_NAME}' not found. Please share it with the service account."

        # Create timestamped tab name if not given
        if not tab_name:
            tab_name = "Search-" + datetime.now().strftime("%Y%m%d-%H%M%S")

        # Sanitize tab name (Google Sheets doesn't allow: [ ] * / \ ? :)
        tab_name = re.sub(r"[\[\]\*/\\?:]", "-", tab_name)[:100]

        # Check if tab exists; if not, create it
        try:
            ws = sh.worksheet(tab_name)
            ws.clear()
        except gspread.WorksheetNotFound:
            ws = sh.add_worksheet(title=tab_name, rows=1000, cols=40)

        safe_df = df.fillna("").astype(str)
        ws.update([safe_df.columns.values.tolist()] + safe_df.values.tolist())

        # Return the URL of this specific tab
        sheet_url = f"https://docs.google.com/spreadsheets/d/{sh.id}/edit#gid={ws.id}"
        return True, f"Saved {len(df)} leads to tab '{tab_name}' → {sheet_url}"
    except Exception as e:
        return False, f"Sheet error: {str(e)[:200]}"


def load_from_sheet():
    ws, err = get_gsheet()
    if ws is None:
        return pd.DataFrame(columns=COLUMNS)
    try:
        data = ws.get_all_values()
        if not data or len(data) < 2:
            return pd.DataFrame(columns=COLUMNS)
        header = data[0]
        rows = data[1:]
        df = pd.DataFrame(rows, columns=header)
        for c in COLUMNS:
            if c not in df.columns:
                df[c] = ""
        return df
    except Exception:
        return pd.DataFrame(columns=COLUMNS)


def build_previous_leads_index():
    seen = {}
    try:
        for fname in os.listdir(HISTORY_DIR):
            if not fname.endswith(".json"): continue
            try:
                with open(os.path.join(HISTORY_DIR, fname), "r", encoding="utf-8") as f:
                    data = json.load(f)
            except Exception: continue
            ts = data.get("timestamp", "")
            context = f"{data.get('industry','')} · {data.get('location','')}"
            for lead in data.get("leads", []):
                n = str(lead.get("Business Name","")).strip().lower()
                w = str(lead.get("Website","")).strip().lower()
                for k in ([f"name:{n}"] if n else []) + ([f"site:{w}"] if w else []):
                    if k not in seen: seen[k] = {"timestamp": ts, "context": context}
    except Exception: pass
    return seen


def mark_duplicates(df, prev_index):
    dups, first = [], []
    for _, row in df.iterrows():
        n = str(row.get("Business Name","")).strip().lower()
        w = str(row.get("Website","")).strip().lower()
        hit = None
        if n and f"name:{n}" in prev_index: hit = prev_index[f"name:{n}"]
        elif w and f"site:{w}" in prev_index: hit = prev_index[f"site:{w}"]
        if hit:
            dups.append("Yes"); first.append(f"{hit['timestamp']} · {hit['context']}")
        else:
            dups.append("No"); first.append("")
    df = df.copy()
    df["Duplicate"] = dups
    df["First Seen"] = first
    return df


def save_history(industry, location, keywords, df):
    try:
        sid = str(uuid.uuid4())[:8]
        ts = datetime.now().strftime("%Y-%m-%d %H:%M")
        data = {"id": sid, "timestamp": ts, "industry": industry,
                "location": location, "keywords": keywords,
                "lead_count": len(df), "leads": df.to_dict(orient="records")}
        with open(os.path.join(HISTORY_DIR, f"{sid}.json"), "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    except Exception: pass


# ============================================================
# HELPERS
# ============================================================
PHONE_REGEX = re.compile(r"(?:\+?1[\s\-.]?)?\(?\b([2-9]\d{2})\)?[\s\-.]?(\d{3})[\s\-.]?(\d{4})\b")
EMAIL_REGEX = re.compile(r"[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}")

JUNK_EMAILS = ["example", "sentry", "yourdomain", "@2x", ".png", ".jpg", ".jpeg",
               ".gif", ".webp", "wixpress", "sentry.io", "w3.org", "domain.com",
               "email.com", "yourname", "noreply", "no-reply", "info@example"]

FB_JUNK_SEGMENTS = {
    "sharer", "sharer.php", "plugins", "login", "login.php", "dialog",
    "tr", "profile.php", "pages", "events", "groups", "photo.php",
    "story.php", "permalink.php", "watch", "marketplace", "gaming",
    "help", "policies", "about", "legal", "careers", "business", "developers"
}


def clean_facebook_url(url):
    if not url: return ""
    try:
        url = url.split("?")[0].split("#")[0]
        p = urlparse(url)
        path = p.path.strip("/")
        if not path: return ""
        first_seg = path.split("/")[0].lower()
        if first_seg in FB_JUNK_SEGMENTS: return ""
        parts = [seg for seg in path.split("/") if seg and seg.lower() not in FB_JUNK_SEGMENTS]
        if not parts: return ""
        handle = parts[0]
        if len(handle) < 3 or len(handle) > 60: return ""
        if "." in handle or " " in handle: return ""
        if handle.lower() in ("home", "welcome", "index", "en", "us"): return ""
        return f"https://www.facebook.com/{handle}"
    except Exception:
        return ""


def clean_linkedin_company_url(url):
    if not url: return ""
    try:
        url = url.split("?")[0].split("#")[0]
        p = urlparse(url)
        path = p.path.strip("/")
        m = re.match(r"company/([A-Za-z0-9\-_%]+)", path, re.IGNORECASE)
        if m:
            h = m.group(1)
            if len(h) > 1: return f"https://www.linkedin.com/company/{h}"
        return ""
    except Exception:
        return ""


def clean_instagram_url(url):
    if not url: return ""
    try:
        url = url.split("?")[0].split("#")[0]
        p = urlparse(url)
        path = p.path.strip("/")
        if not path: return ""
        handle = path.split("/")[0]
        if handle.lower() in ("p", "explore", "reel", "reels", "stories", "accounts"):
            return ""
        if len(handle) < 2 or len(handle) > 40: return ""
        return f"https://www.instagram.com/{handle}"
    except Exception:
        return ""


COUNTRY_HINTS = {
    "united states": ["usa","united states","u.s.a","us"],
    "united kingdom": ["uk","united kingdom","england","scotland","wales","great britain"],
    "canada": ["canada","ca"], "australia": ["australia","au"],
    "india": ["india","in"], "pakistan": ["pakistan","pk"],
    "germany": ["germany","de"], "france": ["france","fr"],
    "spain": ["spain","es"], "italy": ["italy","it"],
    "netherlands": ["netherlands","nl"], "uae": ["uae","dubai","abu dhabi"],
    "saudi arabia": ["saudi"], "singapore": ["singapore","sg"],
    "japan": ["japan","jp"], "china": ["china","cn"],
    "brazil": ["brazil","br"], "mexico": ["mexico","mx"],
    "new zealand": ["new zealand","nz"], "ireland": ["ireland","ie"],
    "south africa": ["south africa","za"],
}

US_STATES = {
    "AL","AK","AZ","AR","CA","CO","CT","DE","FL","GA","HI","ID","IL","IN","IA",
    "KS","KY","LA","ME","MD","MA","MI","MN","MS","MO","MT","NE","NV","NH","NJ",
    "NM","NY","NC","ND","OH","OK","OR","PA","RI","SC","SD","TN","TX","UT","VT",
    "VA","WA","WV","WI","WY","DC"
}


def detect_country(lead):
    try:
        c = str(lead.get("Country", "")).strip()
        if c:
            for known, aliases in COUNTRY_HINTS.items():
                if c.lower() in aliases or c.lower() == known:
                    return known.title()
        state = str(lead.get("State", "")).strip().upper()
        if state in US_STATES: return "United States"
        zipc = str(lead.get("ZIP Code", "")).strip()
        if re.match(r"^\d{5}(-\d{4})?$", zipc): return "United States"
        phone = str(lead.get("Phone Number", "")).strip()
        if phone and phone != "N/A":
            d = re.sub(r"\D", "", phone)
            if d.startswith("1") and len(d) == 11: return "United States"
            if d.startswith("44"): return "United Kingdom"
            if d.startswith("91"): return "India"
            if d.startswith("92"): return "Pakistan"
            if d.startswith("61"): return "Australia"
            if d.startswith("971"): return "UAE"
        website = str(lead.get("Website", "")).strip()
        if website:
            tld = urlparse(website).netloc.rsplit(".", 1)[-1].lower()
            tld_map = {"uk":"United Kingdom","ca":"Canada","au":"Australia","in":"India",
                       "pk":"Pakistan","de":"Germany","fr":"France","es":"Spain","it":"Italy",
                       "nl":"Netherlands","ae":"UAE","sa":"Saudi Arabia","sg":"Singapore",
                       "jp":"Japan","cn":"China","br":"Brazil","mx":"Mexico","nz":"New Zealand",
                       "ie":"Ireland","za":"South Africa"}
            if tld in tld_map: return tld_map[tld]
    except Exception: pass
    return ""


def safe_get(url, timeout=8):
    try:
        return requests.get(url, headers={"User-Agent": HEADERS["User-Agent"]},
                            timeout=timeout, allow_redirects=True)
    except Exception:
        return None


def domain_from_url(url):
    try:
        p = urlparse(url); d = p.netloc.lower()
        if d.startswith("www."): d = d[4:]
        return d
    except Exception: return ""


def extract_jsonld(soup):
    data = {}
    try:
        for script in soup.find_all("script", type="application/ld+json"):
            try:
                obj = json.loads(script.string or "")
            except Exception: continue
            items = obj if isinstance(obj, list) else [obj]
            for item in items:
                if not isinstance(item, dict): continue
                t = item.get("@type", "")
                if isinstance(t, list): t = t[0] if t else ""
                if t in ("LocalBusiness", "Organization", "Store", "MedicalBusiness",
                         "HealthAndBeautyBusiness", "ProfessionalService", "Restaurant",
                         "HomeAndConstructionBusiness", "GeneralContractor",
                         "RoofingContractor", "Electrician", "Plumber", "HousePainter",
                         "MovingCompany", "Locksmith"):
                    addr = item.get("address", {})
                    if isinstance(addr, dict):
                        data.setdefault("Street Address", addr.get("streetAddress", ""))
                        data.setdefault("City", addr.get("addressLocality", ""))
                        data.setdefault("State", addr.get("addressRegion", ""))
                        data.setdefault("ZIP Code", addr.get("postalCode", ""))
                        data.setdefault("Country", addr.get("addressCountry", ""))
                    elif isinstance(addr, str):
                        data.setdefault("Street Address", addr)
                    data.setdefault("Phone Number", item.get("telephone", ""))
                    data.setdefault("Email", item.get("email", ""))
                    geo = item.get("geo", {})
                    if isinstance(geo, dict):
                        lat, lng = geo.get("latitude"), geo.get("longitude")
                        if lat and lng:
                            data.setdefault("Google Maps URL",
                                            f"https://www.google.com/maps?q={lat},{lng}")
                    rating = item.get("aggregateRating", {})
                    if isinstance(rating, dict):
                        data.setdefault("Rating", str(rating.get("ratingValue", "")))
                        data.setdefault("Review Count", str(rating.get("reviewCount", "")))
                    for same in item.get("sameAs", []) or []:
                        low = same.lower()
                        if "facebook.com" in low:
                            cl = clean_facebook_url(same)
                            if cl: data.setdefault("Facebook Page", cl)
                        elif "instagram.com" in low:
                            cl = clean_instagram_url(same)
                            if cl: data.setdefault("Instagram", cl)
                        elif "linkedin.com/company" in low:
                            cl = clean_linkedin_company_url(same)
                            if cl: data.setdefault("LinkedIn Company URL", cl)
                        elif "youtube.com" in low:
                            data.setdefault("YouTube", same)
                        elif "twitter.com" in low or "x.com" in low:
                            data.setdefault("Twitter", same)
    except Exception: pass
    return data


def extract_contacts_from_html(html, base_url):
    result = {}
    soup = BeautifulSoup(html, "lxml")
    for a in soup.find_all("a", href=True):
        h = a["href"]
        if h.startswith("tel:"):
            d = re.sub(r"\D", "", h[4:])
            if len(d) >= 10:
                result.setdefault("Phone Number", h[4:].strip())
        elif h.startswith("mailto:"):
            e = h[7:].split("?")[0].strip()
            if EMAIL_REGEX.match(e) and not any(x in e.lower() for x in JUNK_EMAILS):
                result.setdefault("Email", e)
    for a in soup.find_all("a", href=True):
        h = a["href"]; low = h.lower()
        if "facebook.com" in low or "fb.com" in low:
            cl = clean_facebook_url(h if h.startswith("http") else urljoin(base_url, h))
            if cl: result.setdefault("Facebook Page", cl)
        elif "instagram.com" in low:
            cl = clean_instagram_url(h if h.startswith("http") else urljoin(base_url, h))
            if cl: result.setdefault("Instagram", cl)
        elif "linkedin.com/company" in low:
            cl = clean_linkedin_company_url(h if h.startswith("http") else urljoin(base_url, h))
            if cl: result.setdefault("LinkedIn Company URL", cl)
        elif "youtube.com" in low:
            full = h if h.startswith("http") else urljoin(base_url, h)
            result.setdefault("YouTube", full)
        elif "twitter.com" in low or "x.com" in low:
            full = h if h.startswith("http") else urljoin(base_url, h)
            result.setdefault("Twitter", full)
    addr = soup.find("address")
    if addr:
        t = addr.get_text(separator=" ", strip=True)
        if len(t) > 10: result.setdefault("_address_block", t)
    jsonld = extract_jsonld(soup)
    for k, v in jsonld.items():
        if v and k not in result: result[k] = v
    page_text = soup.get_text(separator=" ", strip=True)
    if "Phone Number" not in result:
        phones = PHONE_REGEX.findall(page_text)
        if phones:
            d = phones[0]; result["Phone Number"] = f"({d[0]}) {d[1]}-{d[2]}"
    if "Email" not in result:
        emails = [e for e in EMAIL_REGEX.findall(page_text)
                  if not any(x in e.lower() for x in JUNK_EMAILS)]
        if emails: result["Email"] = emails[0]
    for a in soup.find_all("a", href=True):
        if "google.com/maps" in a["href"] or "maps.google" in a["href"]:
            result.setdefault("Google Maps URL", a["href"]); break
    return result


# ============================================================
# EXECUTIVE FINDER
# ============================================================
TITLE_KEYWORDS = [
    "owner", "co-owner", "founder", "co-founder", "ceo", "chief executive",
    "president", "managing director", "director", "general manager",
    "partner", "principal", "proprietor", "medical director", "practice manager",
    "manager", "cfo", "coo", "cto", "cmo", "vp", "vice president",
]

BAD_NAME_WORDS = {
    "the", "and", "our", "your", "their", "his", "her", "it", "this", "that",
    "at", "in", "on", "of", "for", "with", "by", "to", "from",
    "construction", "company", "corporation", "corp", "inc", "llc",
    "services", "service", "group", "solution", "solutions",
    "building", "general", "contractor", "contracting", "roofing",
    "plumbing", "electrical", "painting", "concrete", "remodeling",
    "about", "contact", "home", "more", "learn", "click", "here",
    "privacy", "policy", "terms", "cookie", "read", "view", "see",
    "welcome", "hello", "team", "staff", "people", "meet",
    "restaurant", "cafe", "kitchen", "food", "diner", "bar", "grill",
}


def is_valid_name(name):
    if not name: return False
    words = name.strip().split()
    if len(words) < 2 or len(words) > 4: return False
    for w in words:
        lw = w.lower()
        if lw in BAD_NAME_WORDS: return False
        if not re.match(r"^[A-Z][a-zA-Z'\-]+$", w): return False
        if len(w) < 2: return False
    return True


def find_executives_from_html(html, base_url):
    soup = BeautifulSoup(html, "lxml")
    text = soup.get_text(separator="\n", strip=True)
    executives = []
    name_pat = re.compile(
        r"([A-Z][a-z]+(?:\s+[A-Z][a-z'\-]+){1,2})\s*[,\-–|]\s*([A-Za-z\s]{2,40})",
        re.MULTILINE
    )
    for match in name_pat.finditer(text):
        name = match.group(1).strip()
        title = match.group(2).strip()
        if not is_valid_name(name): continue
        for kw in TITLE_KEYWORDS:
            if kw in title.lower():
                executives.append({"name": name, "title": kw.title(), "source": base_url})
                break
        if len(executives) >= 5: break
    seen = set(); result = []
    for e in executives:
        key = e["name"].lower()
        if key not in seen:
            seen.add(key); result.append(e)
    return result


def google_search_executive(business_name, city="", state=""):
    """Improved - searches multiple angles."""
    if not business_name: return {}
    info = {}

    # Query 1: General exec search
    q1 = f'"{business_name}" (CEO OR Founder OR Owner OR President)'
    if city: q1 += f" {city}"
    results = web_search(q1, max_results=8)
    combined = " ".join([f"{r.get('title','')} {r.get('body','')}" for r in results])

    for kw in ["CEO", "Founder", "Co-Founder", "Owner", "President",
               "Managing Director", "Director", "Partner", "Head Chef"]:
        pat = re.compile(
            rf"([A-Z][a-z]+(?:\s+[A-Z][a-z'\-]+){{1,2}})\s*[,\-–]\s*{kw}\b",
            re.IGNORECASE
        )
        m = pat.search(combined)
        if m:
            nm = m.group(1).strip()
            if is_valid_name(nm):
                info["Executive Name"] = nm
                info["Executive Title"] = kw.title()
                break

    # Query 2: LinkedIn profile search
    q2 = f'"{business_name}" site:linkedin.com/in'
    if city: q2 += f" {city}"
    for r in web_search(q2, max_results=5):
        href = r.get("href", "")
        if "linkedin.com/in/" in href:
            info["Executive LinkedIn"] = href
            title = r.get("title", "")
            m = re.match(r"([A-Z][a-z]+\s+[A-Z][a-z'\-]+)", title)
            if m and "Executive Name" not in info:
                nm = m.group(1)
                if is_valid_name(nm):
                    info["Executive Name"] = nm
            break

    # Query 3: Company website "team/about" search
    if "Executive Name" not in info:
        q3 = f'"{business_name}" (team OR about OR leadership OR owners)'
        if city: q3 += f" {city}"
        results3 = web_search(q3, max_results=5)
        for r in results3:
            href = r.get("href", "")
            if href and "linkedin" not in href:
                # Try fetching the page and extracting execs
                resp = safe_get(href, timeout=8)
                if resp and resp.status_code == 200:
                    execs = find_executives_from_html(resp.text, href)
                    if execs:
                        top = execs[0]
                        if is_valid_name(top["name"]):
                            info["Executive Name"] = top["name"]
                            info["Executive Title"] = top["title"]
                            info["Executive Source"] = href
                        break
    return info


def find_company_socials(business_name, city="", state=""):
    result = {}
    if not business_name: return result
    q_fb = f'"{business_name}" facebook'
    if city: q_fb += f" {city}"
    for r in web_search(q_fb, max_results=4):
        url = r.get("href", "")
        if "facebook.com" in url:
            cl = clean_facebook_url(url)
            if cl:
                result["Facebook Page"] = cl
                break
    q_li = f'"{business_name}" site:linkedin.com/company'
    if city: q_li += f" {city}"
    for r in web_search(q_li, max_results=4):
        url = r.get("href", "")
        if "linkedin.com/company" in url:
            cl = clean_linkedin_company_url(url)
            if cl:
                result["LinkedIn Company URL"] = cl
                break
    return result


def detect_email_pattern(domain, business_name):
    if not domain: return "", ""
    q = f'"@{domain}"'
    results = web_search(q, max_results=5)
    combined = " ".join([f"{r.get('title','')} {r.get('body','')}" for r in results])
    emails = [e for e in EMAIL_REGEX.findall(combined)
              if domain in e.lower() and not any(x in e.lower() for x in JUNK_EMAILS)]
    pattern = "{first}"
    if emails:
        local = emails[0].split("@")[0].lower()
        if "." in local and local.count(".") == 1: pattern = "{first}.{last}"
        elif local.isalpha() and len(local) > 3: pattern = "{first}"
        elif len(local) == 1: pattern = "{f}{last}"
    return pattern, emails[0] if emails else ""


def infer_email(pattern, name, domain):
    if not name or not domain or not pattern: return ""
    parts = name.strip().split()
    if len(parts) < 2: return ""
    first = re.sub(r"[^a-z]", "", parts[0].lower())
    last = re.sub(r"[^a-z]", "", parts[-1].lower())
    f = first[0] if first else ""
    try: local = pattern.format(first=first, last=last, f=f)
    except Exception: local = first
    return f"{local}@{domain}"


# ============================================================
# OVERPASS
# ============================================================
OVERPASS_ENDPOINTS = [
    "https://maps.mail.ru/osm/tools/overpass/api/interpreter",
    "https://overpass-api.de/api/interpreter",
    "https://overpass.kumi.systems/api/interpreter",
    "https://overpass.private.coffee/api/interpreter",
]

INDUSTRY_TAG_MAP = {
    "restaurant": [("amenity", "restaurant"), ("amenity", "fast_food")],
    "cafe": [("amenity", "cafe")],
    "coffee": [("amenity", "cafe")],
    "bar": [("amenity", "bar"), ("amenity", "pub")],
    "pub": [("amenity", "pub")],
    "hotel": [("tourism", "hotel")],
    "dentist": [("amenity", "dentist"), ("healthcare", "dentist")],
    "doctor": [("amenity", "doctors"), ("healthcare", "doctor")],
    "hospital": [("amenity", "hospital")],
    "clinic": [("amenity", "clinic")],
    "pharmacy": [("amenity", "pharmacy")],
    "veterinary": [("amenity", "veterinary")],
    "vet": [("amenity", "veterinary")],
    "gym": [("leisure", "fitness_centre")],
    "fitness": [("leisure", "fitness_centre")],
    "yoga": [("leisure", "fitness_centre")],
    "salon": [("shop", "beauty"), ("shop", "hairdresser")],
    "hair": [("shop", "hairdresser")],
    "barber": [("shop", "hairdresser")],
    "spa": [("leisure", "spa"), ("shop", "beauty")],
    "beauty": [("shop", "beauty")],
    "bank": [("amenity", "bank")],
    "supermarket": [("shop", "supermarket")],
    "grocery": [("shop", "supermarket"), ("shop", "convenience")],
    "bakery": [("shop", "bakery")],
    "clothing": [("shop", "clothes")],
    "shoes": [("shop", "shoes")],
    "jewelry": [("shop", "jewelry")],
    "furniture": [("shop", "furniture")],
    "electronics": [("shop", "electronics")],
    "hardware": [("shop", "hardware"), ("shop", "doityourself")],
    "construction": [("office", "company"), ("craft", "builder")],
    "contractor": [("office", "company"), ("craft", "builder")],
    "builder": [("craft", "builder")],
    "plumber": [("craft", "plumber")],
    "electrician": [("craft", "electrician")],
    "roofer": [("craft", "roofer")],
    "painter": [("craft", "painter")],
    "carpenter": [("craft", "carpenter")],
    "lawyer": [("office", "lawyer")],
    "attorney": [("office", "lawyer")],
    "accountant": [("office", "accountant")],
    "insurance": [("office", "insurance")],
    "real estate": [("office", "estate_agent")],
    "architect": [("office", "architect")],
    "engineer": [("office", "engineer")],
    "travel": [("office", "travel_agent")],
    "car repair": [("shop", "car_repair")],
    "auto repair": [("shop", "car_repair")],
    "car dealer": [("shop", "car")],
    "gas": [("amenity", "fuel")],
    "laundry": [("shop", "laundry")],
    "florist": [("shop", "florist")],
    "bookstore": [("shop", "books")],
    "school": [("amenity", "school")],
    "university": [("amenity", "university")],
    "library": [("amenity", "library")],
    "cinema": [("amenity", "cinema")],
    "museum": [("tourism", "museum")],
}

FALLBACK_TAGS = [("amenity", "restaurant")]


def infer_osm_tags(industry):
    if not industry: return FALLBACK_TAGS
    low = industry.lower().strip()
    for key, tags in INDUSTRY_TAG_MAP.items():
        if key in low: return tags
    words = re.split(r"\s+", low)
    for w in words:
        if w in INDUSTRY_TAG_MAP: return INDUSTRY_TAG_MAP[w]
    return FALLBACK_TAGS


def geocode_location(location):
    try:
        url = "https://nominatim.openstreetmap.org/search"
        params = {"q": location, "format": "json", "limit": 1}
        r = requests.get(url, params=params,
                         headers={"User-Agent": "LeadGenApp/1.0"}, timeout=12)
        if r.status_code != 200: return None
        data = r.json()
        if not data: return None
        item = data[0]
        if "boundingbox" in item:
            south, north, west, east = item["boundingbox"]
            return {"type": "bbox", "south": float(south), "north": float(north),
                    "west": float(west), "east": float(east),
                    "display_name": item.get("display_name", location)}
        return None
    except Exception:
        return None


def build_overpass_query(tags, bbox):
    b = f"{bbox['south']},{bbox['west']},{bbox['north']},{bbox['east']}"
    parts = []
    for key, val in tags:
        parts.append(f'node["{key}"="{val}"]({b});')
        parts.append(f'way["{key}"="{val}"]({b});')
        parts.append(f'relation["{key}"="{val}"]({b});')
    body = "\n".join(parts)
    return f"""[out:json][timeout:180];
(
{body}
);
out center tags;"""


def overpass_search(tags, bbox):
    query = build_overpass_query(tags, bbox)
    for endpoint in OVERPASS_ENDPOINTS:
        try:
            r = requests.post(endpoint, data={"data": query},
                              headers={"User-Agent": "LeadGenApp/1.0",
                                       "Content-Type": "application/x-www-form-urlencoded"},
                              timeout=180)
            if r.status_code == 200:
                try:
                    data = r.json()
                    if "elements" in data and len(data["elements"]) > 0:
                        return data["elements"]
                except Exception: continue
        except Exception: continue
    return []


def parse_overpass_element(el, industry):
    tags = el.get("tags", {}) or {}
    name = (
        tags.get("name") or
        tags.get("name:en") or
        tags.get("operator") or
        tags.get("brand") or
        ""
    ).strip()

    if not name:
        return None
    if re.match(r"^\d+$", name):
        return None
    if len(name) < 2:
        return None

    lat = el.get("lat")
    lng = el.get("lon")
    if lat is None or lng is None:
        center = el.get("center") or {}
        lat = center.get("lat"); lng = center.get("lon")

    category = (tags.get("shop") or tags.get("amenity") or tags.get("office") or
                tags.get("craft") or tags.get("leisure") or tags.get("tourism") or
                industry).strip().replace("_", " ").title()

    street_num = (tags.get("addr:housenumber") or "").strip()
    street_name = (tags.get("addr:street") or "").strip()
    street = f"{street_num} {street_name}".strip()
    city = (tags.get("addr:city") or tags.get("addr:town") or tags.get("addr:village") or "").strip()
    state = (tags.get("addr:state") or tags.get("addr:province") or "").strip()
    zipc = (tags.get("addr:postcode") or "").strip()
    country = (tags.get("addr:country") or "").strip()

    phone = (tags.get("phone") or tags.get("contact:phone") or
             tags.get("phone:mobile") or tags.get("contact:mobile") or
             tags.get("mobile") or "").strip()

    website = (tags.get("website") or tags.get("contact:website") or
               tags.get("url") or "").strip()
    if website and not website.startswith("http"):
        website = "https://" + website.lstrip("/")

    email = (tags.get("email") or tags.get("contact:email") or "").strip()
    facebook = (tags.get("contact:facebook") or tags.get("facebook") or "").strip()
    instagram = (tags.get("contact:instagram") or tags.get("instagram") or "").strip()
    twitter = (tags.get("contact:twitter") or tags.get("twitter") or "").strip()
    maps_url = f"https://www.google.com/maps?q={lat},{lng}" if lat and lng else ""

    return {
        "Business Name": name, "Business Category": category,
        "Phone Number": phone, "Email": email, "Website": website,
        "Street Address": street, "City": city, "State": state,
        "ZIP Code": zipc, "Country": country, "Google Maps URL": maps_url,
        "Facebook Page": clean_facebook_url(facebook) if facebook else "",
        "Instagram": clean_instagram_url(instagram) if instagram else "",
        "LinkedIn Company URL": "", "Twitter": twitter, "YouTube": "",
        "Rating": "", "Review Count": "",
        "Source URL": f"https://www.openstreetmap.org/{el.get('type','node')}/{el.get('id','')}",
    }


# ============================================================
# SEARCH
# ============================================================
def ddg_web_search(query, max_results=10):
    results = []
    if not DDGS_OK: return results
    try:
        with DDGS() as ddgs:
            for r in ddgs.text(query, max_results=max_results):
                results.append({"title": r.get("title",""),
                                "href": r.get("href",""),
                                "body": r.get("body","")})
    except Exception: pass
    return results


def google_web_search(query, max_results=10):
    global GOOGLE_BLOCKED_UNTIL
    results = []
    if not GOOGLE_OK: return results
    if time.time() < GOOGLE_BLOCKED_UNTIL: return results
    try:
        for url in google_search(query, num_results=max_results, lang="en",
                                  region="us", sleep_interval=3, advanced=False):
            results.append({"title": "", "href": url, "body": ""})
            if len(results) >= max_results: break
    except Exception as e:
        msg = str(e)
        if "429" in msg or "Too Many Requests" in msg or "sorry/index" in msg:
            GOOGLE_BLOCKED_UNTIL = time.time() + 600
    return results


def bing_web_search(query, max_results=10):
    results = []
    try:
        url = f"https://www.bing.com/search?q={quote_plus(query)}&count={max_results}"
        r = requests.get(url, headers={"User-Agent": HEADERS["User-Agent"]}, timeout=8)
        if r.status_code != 200: return results
        soup = BeautifulSoup(r.text, "lxml")
        for li in soup.select("li.b_algo")[:max_results]:
            a = li.find("a", href=True)
            if not a: continue
            href = a["href"]
            snippet_tag = li.find("p")
            body = snippet_tag.get_text(strip=True) if snippet_tag else ""
            if href.startswith("http"):
                results.append({"title": a.get_text(strip=True), "href": href, "body": body})
    except Exception: pass
    return results


def _throttle():
    global LAST_CALL_TIME
    elapsed = time.time() - LAST_CALL_TIME
    if elapsed < 1.2:
        time.sleep(1.2 - elapsed)
    LAST_CALL_TIME = time.time()


def web_search(query, max_results=10):
    qkey = query.strip().lower()
    if qkey in SEARCH_CACHE:
        return SEARCH_CACHE[qkey][:max_results]
    _throttle()
    results = ddg_web_search(query, max_results)
    if results:
        SEARCH_CACHE[qkey] = results
        return results
    _throttle()
    results = bing_web_search(query, max_results)
    if results:
        SEARCH_CACHE[qkey] = results
        return results
    _throttle()
    results = google_web_search(query, max_results)
    SEARCH_CACHE[qkey] = results
    return results


def call_groq(prompt, json_mode=False, temperature=0.1):
    last_err = None
    for model in GROQ_MODELS:
        try:
            kwargs = {"model": model, "messages": [{"role": "user", "content": prompt}],
                      "temperature": temperature}
            if json_mode: kwargs["response_format"] = {"type": "json_object"}
            return client.chat.completions.create(**kwargs).choices[0].message.content.strip()
        except Exception as e:
            last_err = f"{model}: {e}"; continue
    raise RuntimeError(f"All Groq models failed. Last: {last_err}")


def batch_ai_score(leads, batch_size=25):
    results = []
    for i in range(0, len(leads), batch_size):
        batch = leads[i:i+batch_size]
        lines = []
        for idx, l in enumerate(batch):
            lines.append(
                f"{idx+1}. {l.get('Business Name','')} | {l.get('Business Category','')} | "
                f"{l.get('City','')}, {l.get('State','')} | website:{'yes' if l.get('Website') else 'no'} | "
                f"phone:{'yes' if l.get('Phone Number') else 'no'}"
            )
        listing = "\n".join(lines)
        prompt = f"""Rate each business as a lead from 1 to 10 with a 1-sentence summary (max 20 words).

{listing}

Return ONLY JSON: {{"ratings": [{{"score": 8, "summary": "..."}}, ...]}} in the SAME order."""
        try:
            text = call_groq(prompt, json_mode=True, temperature=0.2)
            text = re.sub(r"^```(json)?", "", text).strip()
            text = re.sub(r"```$", "", text).strip()
            data = json.loads(text)
            ratings = data.get("ratings", [])
            for j in range(len(batch)):
                if j < len(ratings):
                    results.append((str(ratings[j].get("score", "5")),
                                    ratings[j].get("summary", "")))
                else:
                    results.append(("5", ""))
        except Exception:
            for _ in batch:
                results.append(("5", ""))
    while len(results) < len(leads):
        results.append(("5", ""))
    return results[:len(leads)]


# ============================================================
# ENRICHMENT
# ============================================================
CONTACT_PATHS = ["", "/contact", "/contact-us", "/about", "/about-us"]
TEAM_PATHS = ["/team", "/our-team", "/staff", "/leadership", "/management"]


def research_website(url):
    if not url or not url.startswith("http"): return {}
    collected, visited = {}, set()
    base = url.rstrip("/")
    for path in CONTACT_PATHS:
        target = base + path if path else base
        if target in visited: continue
        visited.add(target)
        resp = safe_get(target, timeout=8)
        if not resp or resp.status_code != 200: continue
        info = extract_contacts_from_html(resp.text, target)
        for k, v in info.items():
            if v and k not in collected: collected[k] = v
        time.sleep(0.2)
    return collected


def find_executives(url):
    if not url or not url.startswith("http"): return []
    base = url.rstrip("/")
    all_execs = []
    for path in TEAM_PATHS + ["/about", "/about-us"]:
        target = base + path
        resp = safe_get(target, timeout=8)
        if not resp or resp.status_code != 200: continue
        found = find_executives_from_html(resp.text, target)
        for e in found:
            if not any(x["name"].lower() == e["name"].lower() for x in all_execs):
                all_execs.append(e)
        if len(all_execs) >= 3: break
        time.sleep(0.2)
    return all_execs


def enrich_lead_deep(lead, find_exec=True):
    website = lead.get("Website", "")
    name = lead.get("Business Name", "")
    city = lead.get("City", "")
    state = lead.get("State", "")
    domain = domain_from_url(website)

    site_data = {}
    if website and (not lead.get("Phone Number") or not lead.get("Email")):
        site_data = research_website(website)

    exec_data = {}
    if find_exec and name and website and (not lead.get("Executive Name") or lead.get("Executive Name") in ("", "N/A")):
        site_execs = find_executives(website)
        if site_execs:
            top = site_execs[0]
            exec_data["Executive Name"] = top["name"]
            exec_data["Executive Title"] = top["title"]
            exec_data["Executive Source"] = top["source"]
        google_exec = google_search_executive(name, city, state)
        for k, v in google_exec.items():
            if v and not exec_data.get(k): exec_data[k] = v

    if not lead.get("Facebook Page") or not lead.get("LinkedIn Company URL"):
        socials = find_company_socials(name, city, state)
        for k, v in socials.items():
            if v and not lead.get(k): lead[k] = v

    pattern, _ = detect_email_pattern(domain, name)
    exec_data["Email Pattern"] = pattern

    exec_email = ""
    if exec_data.get("Executive Name") and domain and pattern:
        exec_email = infer_email(pattern, exec_data["Executive Name"], domain)
    if lead.get("Email"):
        exec_data["Executive Email"] = lead["Email"]
    elif exec_email:
        exec_data["Executive Email"] = exec_email + " (inferred)"
    else:
        exec_data["Executive Email"] = "N/A"

    for k, v in site_data.items():
        if k.startswith("_"): continue
        if k in lead and not lead.get(k) and v: lead[k] = v
        elif k not in lead and v: lead[k] = v
    for k, v in exec_data.items():
        if v and not lead.get(k): lead[k] = v

    if "_address_block" in site_data and not lead.get("Street Address"):
        lead["Street Address"] = site_data["_address_block"][:120]

    detected = detect_country(lead)
    if detected: lead["Country"] = detected
    return lead


# ============================================================
# AUDIT
# ============================================================
def audit_website(url):
    result = {"score": 0, "grade": "N/A", "issues": ""}
    if not url or not url.startswith("http"):
        result["issues"] = "No website"; return result
    try:
        t0 = time.time()
        resp = requests.get(url, headers={"User-Agent": HEADERS["User-Agent"]},
                            timeout=10, allow_redirects=True)
        load_time = round(time.time() - t0, 2)
        soup = BeautifulSoup(resp.text, "lxml")
        has_ssl = urlparse(url).scheme == "https"
        has_title = bool(soup.find("title") and soup.find("title").text.strip())
        has_meta = bool(soup.find("meta", attrs={"name": "description"}))
        has_mobile = bool(soup.find("meta", attrs={"name": "viewport"}))
        score, issues = 0, []
        if has_ssl: score += 20
        else: issues.append("No SSL")
        if load_time < 3: score += 20
        elif load_time < 6: score += 10; issues.append(f"Slow ({load_time}s)")
        else: issues.append(f"Very slow ({load_time}s)")
        if has_title: score += 15
        else: issues.append("No title")
        if has_meta: score += 15
        else: issues.append("No meta")
        if has_mobile: score += 15
        else: issues.append("Not mobile")
        score += 15
        score = max(0, min(100, score))
        result["score"] = score
        result["issues"] = "; ".join(issues) if issues else "Good"
        if score >= 90: g = "A"
        elif score >= 75: g = "B"
        elif score >= 60: g = "C"
        elif score >= 40: g = "D"
        else: g = "F"
        result["grade"] = g
    except Exception as e:
        result["issues"] = f"Error: {str(e)[:50]}"
    return result


def run_full_audit_on_df(df):
    sc, gr, iss = [], [], []
    for _, row in df.iterrows():
        w = row.get("Website", "")
        if w and w.startswith("http"):
            a = audit_website(w)
            sc.append(a["score"]); gr.append(a["grade"]); iss.append(a["issues"])
        else:
            sc.append(0); gr.append("N/A"); iss.append("No website")
    df = df.copy()
    df["Audit Score"] = sc
    df["Audit Grade"] = gr
    df["Issues"] = iss
    return df


# ============================================================
# UTILITIES
# ============================================================
def normalize_phone(p):
    if not p or p == "N/A": return p or ""
    d = re.sub(r"\D", "", str(p))
    if len(d) == 10: return f"({d[:3]}) {d[3:6]}-{d[6:]}"
    if len(d) == 11 and d.startswith("1"): return f"({d[1:4]}) {d[4:7]}-{d[7:]}"
    return str(p).strip()


def clean_df(rows):
    for row in rows:
        for c in COLUMNS:
            if c not in row: row[c] = ""
    df = pd.DataFrame(rows, columns=COLUMNS)
    for c in df.columns:
        df[c] = df[c].astype(str).str.strip().replace({"nan": "", "None": ""})
    df["Phone Number"] = df["Phone Number"].apply(normalize_phone)
    df = df[(df["Business Name"] != "") | (df["Website"] != "")]
    df = df.drop_duplicates(subset=["Business Name", "Website"], keep="first")
    return df.reset_index(drop=True)


def ask_ai_about_leads(q, df):
    txt = df.head(30).to_csv(index=False)
    prompt = f"""Lead gen assistant. Data:
---LEADS---
{txt}
---LEADS---
User: "{q}"
Answer concisely."""
    try: return call_groq(prompt, temperature=0.3)
    except Exception as e: return f"Error: {e}"


def generate_email(tname, tinst, tone, lead):
    s = f"""Business: {lead.get('Business Name','')}
Exec: {lead.get('Executive Name','')} - {lead.get('Executive Title','')}
Category: {lead.get('Business Category','')}
City: {lead.get('City','')}, {lead.get('Country','')}
Audit: {lead.get('Audit Score','')}/100
Issues: {lead.get('Issues','')}
Pitch: {lead.get('Pitch Angle','')}"""
    prompt = f"""Sales copywriter. Write email using style "{tname}".
Instructions: {tinst}
Tone: {tone}

Lead:
{s}

RULES:
- Address executive by first name if available.
- Sign "Best regards, [Your Name]".
- Start with "Subject: ..."
- Return ONLY the email."""
    try: return call_groq(prompt, temperature=0.7)
    except Exception as e: return f"Error: {e}"


def generate_pdf_report(df, industry, location, keywords):
    buf = BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=letter,
                            leftMargin=0.6*inch, rightMargin=0.6*inch,
                            topMargin=0.6*inch, bottomMargin=0.6*inch)
    sty = getSampleStyleSheet()
    tstyle = ParagraphStyle("T", parent=sty["Title"], fontSize=24,
                            textColor=colors.HexColor("#FF4B4B"), spaceAfter=20, alignment=1)
    h2 = ParagraphStyle("H2", parent=sty["Heading2"], fontSize=14,
                        textColor=colors.HexColor("#222222"), spaceBefore=12, spaceAfter=6)
    st_ = []
    st_.append(Spacer(1, 1.5*inch))
    st_.append(Paragraph("Lead Generation Report", tstyle))
    st_.append(Spacer(1, 0.3*inch))
    cover = [["Industry:", industry or "-"], ["Location:", location or "-"],
             ["Keywords:", keywords or "-"],
             ["Generated:", datetime.now().strftime("%B %d, %Y %H:%M")],
             ["Total Leads:", str(len(df))]]
    t = Table(cover, colWidths=[2.2*inch, 3.6*inch])
    t.setStyle(TableStyle([("FONTNAME", (0,0), (0,-1), "Helvetica-Bold"),
                           ("FONTSIZE", (0,0), (-1,-1), 11),
                           ("BOTTOMPADDING", (0,0), (-1,-1), 8),
                           ("TOPPADDING", (0,0), (-1,-1), 8)]))
    st_.append(t); st_.append(PageBreak())
    for i, row in df.iterrows():
        st_.append(Paragraph(f"#{i+1} - {row.get('Business Name','')}", h2))
        d = [
            ["Category", str(row.get("Business Category",""))],
            ["Lead Score", str(row.get("Lead Score",""))],
            ["AI Summary", str(row.get("AI Summary",""))],
            ["Executive", f"{row.get('Executive Name','')} - {row.get('Executive Title','')}"],
            ["Phone", str(row.get("Phone Number",""))],
            ["Email", str(row.get("Email",""))],
            ["Website", str(row.get("Website",""))],
            ["Address", f"{row.get('Street Address','')} {row.get('City','')} {row.get('State','')}".strip()],
            ["Country", str(row.get("Country",""))],
        ]
        t = Table(d, colWidths=[1.4*inch, 5.5*inch])
        t.setStyle(TableStyle([("FONTNAME", (0,0), (0,-1), "Helvetica-Bold"),
                               ("FONTSIZE", (0,0), (-1,-1), 8.5),
                               ("VALIGN", (0,0), (-1,-1), "TOP"),
                               ("BOTTOMPADDING", (0,0), (-1,-1), 3)]))
        st_.append(t); st_.append(Spacer(1, 0.2*inch))
    doc.build(st_); buf.seek(0)
    return buf.getvalue()


# ============================================================
# UI
# ============================================================
st.set_page_config(page_title="Lead Generation Assistant", layout="wide", page_icon="🎯")

if "theme" not in st.session_state:
    st.session_state["theme"] = "Dark"

if "df" not in st.session_state:
    loaded = load_from_sheet()
    if not loaded.empty:
        st.session_state["df"] = loaded

with st.sidebar:
    st.markdown("### Theme")
    tc = st.radio("Theme", ["Dark","Light","System"],
                  index=["Dark","Light","System"].index(st.session_state["theme"]),
                  horizontal=True, label_visibility="collapsed")
    st.session_state["theme"] = tc
    if tc == "Light":
        st.markdown("""<style>
            .stApp { background-color: #fff; color: #111; }
            section[data-testid="stSidebar"] { background-color: #f4f4f9; }
            h1,h2,h3,h4,h5,h6,p,label,span { color: #111 !important; }
            </style>""", unsafe_allow_html=True)
    st.markdown("---")
    st.markdown("## Cloud Sync")

    # Show secret status
    if GCP_JSON_B64:
        st.success(f"✓ GCP secret loaded ({len(GCP_JSON_B64)} chars)")
    else:
        st.error("✗ GCP secret missing")

    if st.button("Save to Sheet", width='stretch'):
        if "df" in st.session_state:
            ok, msg = save_to_sheet(st.session_state["df"])
            if ok: st.success(msg)
            else: st.error(msg)
        else:
            st.warning("No data.")
        st.markdown("---")
    st.markdown("## 🕘 Search History")

    # Load all history files
    hist_items = []
    try:
        for fname in os.listdir(HISTORY_DIR):
            if not fname.endswith(".json"): continue
            try:
                with open(os.path.join(HISTORY_DIR, fname), "r", encoding="utf-8") as f:
                    data = json.load(f)
                hist_items.append(data)
            except Exception: continue
        hist_items.sort(key=lambda x: x.get("timestamp", ""), reverse=True)
    except Exception:
        pass

    if not hist_items:
        st.caption("No history yet.")
    else:
        st.caption(f"{len(hist_items)} past searches")

        if st.button("🗑️ Clear All", width='stretch'):
            try:
                for fname in os.listdir(HISTORY_DIR):
                    if fname.endswith(".json"):
                        os.remove(os.path.join(HISTORY_DIR, fname))
                st.success("Cleared!")
                st.rerun()
            except Exception:
                pass

        for item in hist_items[:20]:
            label = f"📌 {item.get('industry', '')} · {item.get('location', '')}"
            with st.expander(label):
                st.caption(f"🕐 {item.get('timestamp', '')}")
                st.caption(f"👥 {item.get('lead_count', 0)} leads")
                colA, colB = st.columns(2)
                with colA:
                    if st.button("📂 Load", key=f"load_{item['id']}", width='stretch'):
                        st.session_state["df"] = pd.DataFrame(item["leads"], columns=COLUMNS)
                        st.success(f"Loaded {len(item['leads'])} leads")
                        st.rerun()
                with colB:
                    if st.button("🗑️ Delete", key=f"del_{item['id']}", width='stretch'):
                        try:
                            os.remove(os.path.join(HISTORY_DIR, f"{item['id']}.json"))
                            st.rerun()
                        except Exception:
                            pass
st.title("🎯 Lead Generation Assistant")
st.caption("OpenStreetMap · Groq AI · Website enrichment · Cloud sync")

c1, c2, c3, c4 = st.columns([2, 2, 1, 2])
with c1: industry = st.text_input("Industry", value="Restaurants")
with c2: location = st.text_input("Location", value="San Francisco, California")
with c3: num_leads = st.number_input("Number of Leads", min_value=5, max_value=500, value=50)
with c4: keywords = st.text_input("Keywords (optional)", placeholder="e.g. Botox, Laser")

deep_research = st.checkbox("🔬 Deep research (adds socials, emails, website scrape)", value=False)
search_executives = st.checkbox("👤 Also find executives (slower)", value=False)

if st.button("Find Leads", type="primary"):
    if not industry or not location:
        st.warning("Please fill in Industry and Location."); st.stop()

    ph_s, ph_e, ph_r = st.empty(), st.empty(), st.empty()
    prog = st.progress(0)

    ph_s.info(f"Geocoding location: {location}...")
    bbox = geocode_location(location)
    if not bbox:
        st.error(f"Could not find location: '{location}'")
        st.stop()
    ph_s.success(f"Location found: {bbox['display_name'][:80]}")
    prog.progress(10)

    tags = infer_osm_tags(industry)
    ph_s.info(f"Querying OpenStreetMap for '{industry}'...")
    elements = overpass_search(tags, bbox)
    if not elements:
        st.error("No businesses found.")
        st.stop()

    leads = []
    for el in elements:
        p = parse_overpass_element(el, industry)
        if p: leads.append(p)

    seen = set(); unique = []
    for l in leads:
        key = l.get("Business Name", "").strip().lower()
        if key and key not in seen:
            seen.add(key); unique.append(l)

    ph_s.success(f"Found {len(unique)} unique businesses.")
    prog.progress(30)

    unique = unique[:num_leads * 2]

    ph_e.info(f"AI scoring {len(unique)} leads...")
    scores = batch_ai_score(unique, batch_size=25)
    for i, (score, summary) in enumerate(scores):
        unique[i]["Lead Score"] = score
        unique[i]["AI Summary"] = summary
    ph_e.success("AI scoring complete.")
    prog.progress(45)

    for l in unique:
        l.setdefault("Outreach Status", "New")

    df = clean_df(unique)

    er = df.to_dict(orient="records")
    if deep_research:
        to_enrich = min(len(er), num_leads)
        ph_r.info(f"🔬 Deep research on {to_enrich} leads (parallel)...")

        completed = [0]
        enriched_results = list(er)

        def _enrich_one(idx_lead):
            idx, lead = idx_lead
            try:
                return idx, enrich_lead_deep(lead, find_exec=search_executives)
            except Exception:
                return idx, lead

        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(_enrich_one, (i, er[i])) for i in range(to_enrich)]
            for fut in concurrent.futures.as_completed(futures):
                try:
                    i, res = fut.result()
                    enriched_results[i] = res
                except Exception:
                    pass
                completed[0] += 1
                prog.progress(45 + int((completed[0] / to_enrich) * 45))
                ph_r.info(f"🔬 Deep research: {completed[0]}/{to_enrich} done")

        er = enriched_results
        prog.empty(); ph_r.empty()
    else:
        ph_r.success("⚡ Fast mode: skipping deep research")
        time.sleep(1)
        prog.empty(); ph_r.empty()

    df = clean_df(er)
    df = df.head(num_leads).reset_index(drop=True)

    for col in ["Executive Name", "Executive Title", "Executive Email", "Executive LinkedIn"]:
        if col in df.columns:
            df[col] = df[col].replace("", "N/A")

    if "Outreach Status" not in df.columns:
        df["Outreach Status"] = "New"
    df["Outreach Status"] = df["Outreach Status"].replace("", "New")

    pi = build_previous_leads_index()
    df = mark_duplicates(df, pi)
    dc = (df["Duplicate"] == "Yes").sum()
    if dc > 0:
        st.info(f"{dc} duplicate(s) found")

    st.session_state["df"] = df
    save_history(industry, location, keywords, df)
    tab_name = f"{industry[:20]}-{location[:25]}".replace("/", "-").replace("\\", "-")

    with st.spinner(f"Saving to new tab '{tab_name}'..."):
        ok, msg = save_to_sheet(df, tab_name=tab_name)
    if ok:
        st.success(f"☁️ {msg}")
        if "→" in msg:
            sheet_url = msg.split("→")[-1].strip()
            st.markdown(f"[**📊 Open Google Sheet tab →**]({sheet_url})")
    else:
        st.warning(f"⚠️ Cloud sync: {msg}")


if "df" in st.session_state:
    df = st.session_state["df"]
    st.success(f"Found {len(df)} businesses")

    st.markdown("---")
    st.markdown("### 📞 Outreach Status")
    status_col1, status_col2 = st.columns([2, 3])
    with status_col1:
        lead_names = df["Business Name"].tolist()
        sel_lead = st.selectbox("Select lead", lead_names, key="outreach_lead_sel")
    with status_col2:
        current_status = df[df["Business Name"] == sel_lead]["Outreach Status"].iloc[0] if not df.empty else "New"
        if current_status not in OUTREACH_OPTIONS:
            current_status = "New"
        new_status = st.radio("Status", OUTREACH_OPTIONS,
                              index=OUTREACH_OPTIONS.index(current_status),
                              horizontal=True, key="outreach_status_radio")
        if new_status != current_status:
            df.loc[df["Business Name"] == sel_lead, "Outreach Status"] = new_status
            st.session_state["df"] = df
            ok, msg = save_to_sheet(df)
            if ok:
                st.success(f"Status set to '{new_status}'")
            else:
                st.warning(f"Saved locally. {msg}")

    st.markdown("#### Filter")
    fc, _ = st.columns([1, 3])
    with fc:
        vm = st.selectbox("Show:", ["All leads", "Only new", "Only duplicates"], key="filter_mode")
    vdf = df.copy()
    if vm == "Only new": vdf = vdf[vdf["Duplicate"] == "No"]
    elif vm == "Only duplicates": vdf = vdf[vdf["Duplicate"] == "Yes"]
    st.caption(f"Showing {len(vdf)} of {len(df)}")

    priority = ["Business Name", "Outreach Status", "Duplicate", "Lead Score",
                "Phone Number", "Email", "Website", "Street Address", "City",
                "State", "ZIP Code", "Country", "Google Maps URL",
                "Executive Name", "Executive Title", "Executive Email", "Executive LinkedIn",
                "Facebook Page", "Instagram", "LinkedIn Company URL",
                "Audit Grade", "Audit Score"]
    priority = [c for c in priority if c in vdf.columns]
    others = [c for c in vdf.columns if c not in priority]
    st.dataframe(vdf[priority + others], width='stretch')

    st.markdown("---"); st.markdown("### Website Quality Audit")
    if st.button("Run Website Audit", type="primary"):
        with st.spinner("Auditing..."):
            da = run_full_audit_on_df(df)
        st.session_state["df"] = da
        save_to_sheet(da)
        st.success("Audit complete!")
        st.rerun()

    st.markdown("---"); st.markdown("### Email Generator")
    ec1, ec2, ec3 = st.columns([3, 2, 2])
    with ec1:
        names = vdf["Business Name"].tolist() if not vdf.empty else df["Business Name"].tolist()
        sel_name = st.selectbox("Lead", names, key="email_lead_select")
    with ec2: sel_tpl = st.selectbox("Template", list(EMAIL_TEMPLATES.keys()), key="email_template_select")
    with ec3: sel_tone = st.selectbox("Tone", TONES, key="email_tone_select")
    if st.button("Generate Email", type="primary"):
        row = df[df["Business Name"] == sel_name].iloc[0].to_dict()
        with st.spinner("Writing..."):
            st.session_state["generated_email"] = generate_email(sel_tpl, EMAIL_TEMPLATES[sel_tpl], sel_tone, row)
    if st.session_state.get("generated_email"):
        st.markdown("#### Generated Email")
        ed = st.text_area("Edit:", value=st.session_state["generated_email"], height=300)
        st.text_area("📋 Copy below (Ctrl+A, Ctrl+C):", value=ed, height=100, key="copy_area")

        # --- EXPORT ---
    st.markdown("---"); st.markdown("### 📤 Export")

    ec1, ec2, ec3, ec4 = st.columns(4)

    with ec1:
        st.download_button("📥 CSV", vdf.to_csv(index=False).encode(),
                           "leads.csv", "text/csv", width='stretch')

    with ec2:
        try:
            buf = BytesIO()
            with pd.ExcelWriter(buf, engine="openpyxl") as w:
                vdf.to_excel(w, index=False, sheet_name="Leads")
            st.download_button("📥 Excel", buf.getvalue(), "leads.xlsx",
                "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                width='stretch')
        except Exception as e:
            st.warning(f"Excel: {e}")

    with ec3:
        # Google Sheet direct link + push button
        sheet_url = "https://docs.google.com/spreadsheets/d/1AjfsCTB8zrgiQo0OWjVaRwC_ENjjzPV0i1hQ8GzWkKc/edit"
        if st.button("📊 Open Google Sheet", width='stretch'):
            # Push current data first, then open
            ok, msg = save_to_sheet(vdf)
            if ok:
                st.success("Synced! Opening...")
            st.markdown(f'<meta http-equiv="refresh" content="0; url={sheet_url}">',
                       unsafe_allow_html=True)
            st.markdown(f"[**Click here if it doesn't open →**]({sheet_url})")

    with ec4:
        if st.button("📋 Copy", width='stretch'):
            st.session_state["show_copy"] = not st.session_state.get("show_copy", False)

    if st.session_state.get("show_copy"):
        st.code(vdf.to_csv(index=False), language="text")

    st.markdown("---"); st.markdown("### 💬 Chat with Your Leads")
    if "chat_history" not in st.session_state:
        st.session_state["chat_history"] = []
    for m in st.session_state["chat_history"]:
        with st.chat_message(m["role"]): st.markdown(m["content"])
    uq = st.chat_input("Ask about these leads...")
    if uq:
        st.session_state["chat_history"].append({"role":"user","content":uq})
        with st.chat_message("user"): st.markdown(uq)
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                ans = ask_ai_about_leads(uq, df)
            st.markdown(ans)
        st.session_state["chat_history"].append({"role":"assistant","content":ans})