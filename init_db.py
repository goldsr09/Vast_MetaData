import sqlite3

DB_PATH = 'vast_ads.db'

CREATE_TABLE_SQL = '''
CREATE TABLE IF NOT EXISTS vast_ads (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    call_number INTEGER,
    ad_id TEXT,
    creative_id TEXT,
    title TEXT,
    duration TEXT,
    clickthrough TEXT,
    media_urls TEXT,
    channel_name TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
)
'''



def main():
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute(CREATE_TABLE_SQL)
    conn.commit()
    conn.close()
    print("Table 'vast_ads' has been created (or already exists).")

if __name__ == "__main__":
    main()
