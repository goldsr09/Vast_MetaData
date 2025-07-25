import sqlite3
import requests
from lxml import etree
import json
from urllib.parse import urlparse, parse_qs
import hashlib

DB_PATH = 'vast_ads.db'

CREATE_TABLE_SQL = '''
CREATE TABLE IF NOT EXISTS vast_ads (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    call_number INTEGER,
    ad_id TEXT,
    creative_id TEXT,
    ssai_creative_id TEXT,
    title TEXT,
    duration TEXT,
    clickthrough TEXT,
    media_urls TEXT,
    channel_name TEXT,
    adomain TEXT,
    creative_hash TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    ad_xml TEXT,
    wrapped_ad INTEGER DEFAULT 0,
    initial_metadata_json TEXT
)
'''

def setup_db():
    conn = sqlite3.connect(DB_PATH, timeout=30, isolation_level=None, check_same_thread=False)
    cur = conn.cursor()
    cur.execute(CREATE_TABLE_SQL)
    conn.commit()
    conn.close()

def make_creative_hash(*fields):
    base = ':'.join([str(f) if f else '' for f in fields])
    return hashlib.sha256(base.encode('utf-8')).hexdigest()

def get_ssai_creative_id(ad_element):
    ssai = ad_element.xpath('.//Extensions/Extension[@type="FreeWheel"]/SSAICreativeId')
    if ssai and ssai[0].text:
        value = ssai[0].text.strip()
        # Remove CDATA if present
        if value.startswith("CDATA[") and value.endswith("]"):
            value = value[6:-1]
        return value
    return None

def fetch_and_parse_vast(url, headers, max_depth=5, visited=None, is_wrapped=False):
    import json as _json
    if visited is None:
        visited = set()
    if url in visited or max_depth <= 0:
        return [], None, None
    visited.add(url)
    try:
        response = requests.get(url, headers=headers, timeout=10)
    except Exception:
        return [], None, None
    if response.status_code != 200 or not response.content.strip():
        return [], None, None
    parser = etree.XMLParser(recover=True)
    try:
        tree = etree.fromstring(response.content, parser=parser)
    except etree.XMLSyntaxError:
        return [], None, None
    ads = tree.xpath("//Ad")
    initial_metadata = []
    for ad in ads:
        ad_id = ad.get("id", "N/A")
        title = ad.xpath(".//AdTitle/text()")
        duration = ad.xpath(".//Duration/text()")
        click_url = ad.xpath(".//ClickThrough/text()")
        creative_id = ad.xpath(".//Creative/@id")
        creative_id = creative_id[0] if creative_id else None
        media_files = ad.xpath(".//MediaFile")
        media_urls = [mf.text.strip() for mf in media_files if mf.text]
        ssai_creative_id = get_ssai_creative_id(ad)
        adomain = None
        adomain_nodes = ad.xpath('.//AdVerifications/Verification/AdVerificationParameters/Adomain/text()')
        if not adomain_nodes:
            adomain_nodes = ad.xpath('.//Extension[@type="advertiser"]/Adomain/text()')
        if not adomain_nodes:
            adomain_nodes = ad.xpath('.//Advertiser/text()')
        adomain = adomain_nodes[0] if adomain_nodes else None
        creative_hash = make_creative_hash(ssai_creative_id, creative_id, ','.join(media_urls), adomain if adomain else '')
        meta = {
            "ad_id": ad_id,
            "creative_id": creative_id,
            "ssai_creative_id": ssai_creative_id,
            "title": title[0] if title else None,
            "duration": duration[0] if duration else None,
            "clickthrough": click_url[0] if click_url else None,
            "media_urls": media_urls,
            "adomain": adomain,
            "creative_hash": creative_hash
        }
        initial_metadata.append(meta)
    final_ads = []
    for idx, ad in enumerate(ads):
        wrapper = ad.find("Wrapper")
        if wrapper is not None:
            vast_ad_tag_uri = wrapper.findtext("VASTAdTagURI")
            if vast_ad_tag_uri:
                child_ads, _, child_initial = fetch_and_parse_vast(vast_ad_tag_uri.strip(), headers, max_depth-1, visited, is_wrapped=True)
                for c in child_ads:
                    final_ads.append((c[0], True, _json.dumps(initial_metadata[idx] if idx < len(initial_metadata) else {})))
        else:
            final_ads.append((ad, is_wrapped, _json.dumps(initial_metadata[idx] if idx < len(initial_metadata) else {})))
    return final_ads, response.content.decode(errors='replace'), initial_metadata

def parse_vast_and_store(url, call_number):
    conn = sqlite3.connect(DB_PATH, timeout=30, isolation_level=None, check_same_thread=False)
    cur = conn.cursor()

    headers = {
        "User-Agent": "Roku/DVP-14.5 (14.5.4.5934-46)"
    }

    parsed_url = urlparse(url)
    query_params = parse_qs(parsed_url.query)
    csid = query_params.get("csid", [""])[0]
    csid_parts = csid.split("/")
    channel_name = csid_parts[1] if len(csid_parts) >= 2 else None

    ads, last_xml, _ = fetch_and_parse_vast(url, headers)
    if not ads:
        conn.close()
        return f"❌ No valid Inline ads found."

    for ad, wrapped_flag, initial_metadata_json in ads:
        ad_id = ad.get("id", "N/A")
        title = ad.xpath(".//AdTitle/text()")
        duration = ad.xpath(".//Duration/text()")
        click_url = ad.xpath(".//ClickThrough/text()")
        creative_id = ad.xpath(".//Creative/@id")
        creative_id = creative_id[0] if creative_id else None
        media_files = ad.xpath(".//MediaFile")
        media_urls = [mf.text.strip() for mf in media_files if mf.text]

        ssai_creative_id = get_ssai_creative_id(ad)

        adomain = None
        adomain_nodes = ad.xpath('.//AdVerifications/Verification/AdVerificationParameters/Adomain/text()')
        if not adomain_nodes:
            adomain_nodes = ad.xpath('.//Extension[@type="advertiser"]/Adomain/text()')
        if not adomain_nodes:
            adomain_nodes = ad.xpath('.//Advertiser/text()')
        adomain = adomain_nodes[0] if adomain_nodes else None

        # --- NEW: If no adomain, follow clickthrough and get domain ---
        if adomain is None and click_url and click_url[0]:
            try:
                resp = requests.get(click_url[0], headers=headers, timeout=5, allow_redirects=True)
                final_url = resp.url
                adomain = urlparse(final_url).netloc
            except Exception:
                adomain = None
        # -------------------------------------------------------------

        creative_hash = make_creative_hash(ssai_creative_id, creative_id, ','.join(media_urls), adomain if adomain else '')

        ad_xml = etree.tostring(ad, pretty_print=True, encoding='unicode')

        cur.execute("""
            INSERT INTO vast_ads (
                call_number, ad_id, creative_id, ssai_creative_id, title, duration, clickthrough, media_urls,
                channel_name, adomain, creative_hash, ad_xml, wrapped_ad, initial_metadata_json
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            call_number,
            ad_id,
            creative_id,
            ssai_creative_id,
            title[0] if title else None,
            duration[0] if duration else None,
            click_url[0] if click_url else None,
            json.dumps(media_urls),
            channel_name,
            adomain,
            creative_hash,
            ad_xml,
            int(wrapped_flag),
            initial_metadata_json
        ))

    conn.commit()
    conn.close()
    return f"✅ Parsed and stored {len(ads)} ads."

# Ensure table exists at import
setup_db()
