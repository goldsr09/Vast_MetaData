import sqlite3, json, csv
from datetime import datetime

db_path = 'vast_ads.db'
out_path = 'creative_id_dataset.csv'

schema = [
    'initial_creative_id','final_creative_id','ad_id','ssai_creative_id','wrapper_count','wrapper_chain','adomain','title','duration','clickthrough','media_urls','timestamp','error_flag','wrapper_metadata','creative_hash','position_in_pod','ad_xml'
]

def extract_dataset():
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    # Get all ads, including wrapper info
    cur.execute('''
        SELECT ad_id, creative_id, ssai_creative_id, title, duration, clickthrough, media_urls, adomain, creative_hash, created_at, ad_xml, wrapped_ad, initial_metadata_json
        FROM vast_ads
        ORDER BY id
    ''')
    rows = cur.fetchall()
    dataset = []
    for row in rows:
        (
            ad_id, creative_id, ssai_creative_id, title, duration, clickthrough, media_urls, adomain, creative_hash, created_at, ad_xml, wrapped_ad, initial_metadata_json
        ) = row
        # Parse initial metadata and wrapper chain
        try:
            initial_meta = json.loads(initial_metadata_json) if initial_metadata_json else {}
        except Exception:
            initial_meta = {}
        # Advanced wrapper chain extraction
        wrapper_chain = []
        wrapper_domains = []
        wrapper_metadata = []
        if initial_metadata_json:
            try:
                meta = json.loads(initial_metadata_json)
                if isinstance(meta, dict):
                    wrapper_chain.append(meta.get('creative_id'))
                    wrapper_domains.append(meta.get('adomain'))
                    wrapper_metadata.append(meta)
                elif isinstance(meta, list):
                    for m in meta:
                        wrapper_chain.append(m.get('creative_id'))
                        wrapper_domains.append(m.get('adomain'))
                        wrapper_metadata.append(m)
            except Exception:
                pass
        # Remove Nones and deduplicate
        wrapper_chain = [w for w in wrapper_chain if w]
        wrapper_domains = [w for w in wrapper_domains if w]
        wrapper_count = len(wrapper_chain)
        initial_creative_id = wrapper_chain[0] if wrapper_chain else ''
        wrapper_chain_str = '|'.join(wrapper_chain)
        wrapper_metadata_json = json.dumps(wrapper_metadata)
        error_flag = ''
        position_in_pod = ''
        final_creative_id = creative_id
        dataset.append([
            initial_creative_id,
            final_creative_id,
            ad_id,
            ssai_creative_id,
            wrapper_count,
            wrapper_chain_str,
            adomain,
            title,
            duration,
            clickthrough,
            media_urls,
            created_at,
            error_flag,
            wrapper_metadata_json,
            creative_hash,
            position_in_pod,
            ad_xml
        ])
    conn.close()
    # Write to CSV
    with open(out_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(schema)
        writer.writerows(dataset)
    print(f"Extracted {len(dataset)} rows to {out_path}")

if __name__ == '__main__':
    extract_dataset()
