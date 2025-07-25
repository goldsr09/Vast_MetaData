
import sqlite3
conn = sqlite3.connect('vast_ads.db')
cur = conn.cursor()
cur.execute("PRAGMA table_info(vast_ads)")
print(cur.fetchall())
conn.close()
