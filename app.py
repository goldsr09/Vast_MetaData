
import os
from flask import Flask, request, render_template_string, send_file, redirect, url_for
from parser_1 import parse_vast_and_store

app = Flask(__name__)
# Route to download the entire SQLite database file
@app.route('/export_db')
def export_db():
    db_path = 'vast_ads.db'
    if not os.path.exists(db_path):
        return 'Database file not found.', 404
    return send_file(db_path, as_attachment=True, download_name='vast_ads.db', mimetype='application/octet-stream')

HTML_FORM = '''
<!doctype html>
<html>
  <body>
    <h2>VAST URL Parser</h2>
    <form method="POST">
      <label for="url">Enter VAST Tag URL:</label><br>
      <input type="text" name="url" size="100"><br><br>
      <input type="submit" value="Parse">
    </form>
    <br>
    <a href="/multi">Parse same VAST multiple times</a> |
    <a href="/results">See recent results</a>
    {% if result %}
      <p><strong>Result:</strong> {{ result|safe }}</p>
    {% endif %}
  </body>
</html>
'''

@app.route('/', methods=['GET', 'POST'])
def index():
    
    
    result = None
    if request.method == 'POST':
        url = request.form['url']
        result = parse_vast_and_store(url, call_number=1)
    return render_template_string('''
    <style>
      body { font-family: 'Segoe UI', Arial, sans-serif; background: #f4f6fa; margin: 0; padding: 0; }
      .container { max-width: 700px; margin: 60px auto; background: #fff; border-radius: 12px; box-shadow: 0 2px 12px #0001; padding: 32px 40px 40px 40px; }
      h2 { margin-top: 0; font-size: 2.2em; letter-spacing: 1px; color: #2a3b4c; }
      form { margin-bottom: 24px; }
      label { font-weight: 500; color: #2a3b4c; }
      input[type="text"] { width: 100%; padding: 10px; border-radius: 6px; border: 1px solid #d0d6e2; font-size: 1.1em; margin-top: 6px; }
      input[type="submit"] { background: #2a7be4; color: #fff; border: none; border-radius: 6px; padding: 10px 28px; font-size: 1.1em; font-weight: 500; cursor: pointer; transition: background 0.2s; }
      input[type="submit"]:hover { background: #174a7c; }
      .nav-links { margin-bottom: 18px; }
      .nav-links a { display: inline-block; margin-right: 18px; padding: 7px 18px; border-radius: 6px; background: #e8eefa; color: #2a3b4c; text-decoration: none; font-weight: 500; transition: background 0.2s; }
      .nav-links a:hover { background: #2a3b4c; color: #fff; }
      .result-msg { margin-top: 24px; font-size: 1.1em; color: #174a7c; background: #e8eefa; border-radius: 6px; padding: 12px 18px; }
    </style>
    <div class="container">
      <h2>VAST URL Parser</h2>
      <form method="POST">
        <label for="url">Enter VAST Tag URL:</label><br>
        <input type="text" name="url" size="100"><br><br>
        <input type="submit" value="Parse">
      </form>
      <div class="nav-links">
        <a href="/multi">Parse same VAST multiple times</a>
        <a href="/results">See recent results</a>
      </div>
      {% if result %}
        <div class="result-msg"><strong>Result:</strong> {{ result|safe }}</div>
      {% endif %}
    </div>
    ''', result=result)

@app.route('/multi', methods=['GET', 'POST'])
def multi():
    result = None
    if request.method == 'POST':
        url = request.form['url']
        num_calls = int(request.form.get('num_calls', 3))
        messages = []
        for i in range(num_calls):
            msg = parse_vast_and_store(url, call_number=i+1)
            messages.append(f"Call {i+1}: {msg}")
        result = "<br>".join(messages)
    return render_template_string('''
    <style>
      body { font-family: 'Segoe UI', Arial, sans-serif; background: #f4f6fa; margin: 0; padding: 0; }
      .container { max-width: 700px; margin: 60px auto; background: #fff; border-radius: 12px; box-shadow: 0 2px 12px #0001; padding: 32px 40px 40px 40px; }
      h2 { margin-top: 0; font-size: 2.2em; letter-spacing: 1px; color: #2a3b4c; }
      form { margin-bottom: 24px; }
      label { font-weight: 500; color: #2a3b4c; }
      input[type="text"], input[type="number"] { width: 100%; padding: 10px; border-radius: 6px; border: 1px solid #d0d6e2; font-size: 1.1em; margin-top: 6px; }
      input[type="submit"] { background: #2a7be4; color: #fff; border: none; border-radius: 6px; padding: 10px 28px; font-size: 1.1em; font-weight: 500; cursor: pointer; transition: background 0.2s; }
      input[type="submit"]:hover { background: #174a7c; }
      .nav-links { margin-bottom: 18px; }
      .nav-links a { display: inline-block; margin-right: 18px; padding: 7px 18px; border-radius: 6px; background: #e8eefa; color: #2a3b4c; text-decoration: none; font-weight: 500; transition: background 0.2s; }
      .nav-links a:hover { background: #2a3b4c; color: #fff; }
      .result-msg { margin-top: 24px; font-size: 1.1em; color: #174a7c; background: #e8eefa; border-radius: 6px; padding: 12px 18px; }
    </style>
    <div class="container">
      <h2>Run VAST Tag Multiple Times</h2>
      <form method="POST">
        <label for="url">VAST Tag URL:</label><br>
        <input type="text" name="url" size="100"><br>
        <label for="num_calls">Number of times to run:</label>
        <input type="number" name="num_calls" min="1" max="20" value="3"><br><br>
        <input type="submit" value="Parse Multiple">
      </form>
      <div class="nav-links">
        <a href="/">Back to Single Parse</a>
        <a href="/results">See recent results</a>
      </div>
      {% if result %}
        <div class="result-msg"><strong>Result:</strong><br>{{ result|safe }}</div>
      {% endif %}
    </div>
    ''', result=result)


import sqlite3
import csv
import io
import json

@app.route('/results', methods=['GET', 'POST'])
def results():
    # Advanced filtering/search
    sort = request.args.get('sort', 'id')
    order = request.args.get('order', 'desc')
    page = int(request.args.get('page', 1))
    per_page = 50
    offset = (page - 1) * per_page
    # Multi-field filters
    filters = {}
    for f in ['ad_id','creative_id','ssai_creative_id','title','duration','clickthrough','adomain','creative_hash']:
        v = request.args.get(f, '').strip()
        if v:
            filters[f] = v
    global_search = request.args.get('q', '').strip()
    allowed_sorts = ['id', 'call_number', 'ad_id', 'creative_id', 'ssai_creative_id', 'title', 'duration', 'clickthrough', 'media_urls', 'adomain', 'creative_hash', 'created_at', 'wrapped_ad']
    if sort not in allowed_sorts:
        sort = 'id'
    if order not in ['asc', 'desc']:
        order = 'desc'
    where = []
    params = []
    for k, v in filters.items():
        where.append(f"{k} LIKE ?")
        params.append(f"%{v}%")
    if global_search:
        search_fields = ['ad_id','creative_id','ssai_creative_id','title','duration','clickthrough','media_urls','adomain','creative_hash']
        where.append('(' + ' OR '.join([f"{f} LIKE ?" for f in search_fields]) + ')')
        params.extend([f"%{global_search}%"]*len(search_fields))
    where_clause = f"WHERE {' AND '.join(where)}" if where else ''
    conn = sqlite3.connect('vast_ads.db')
    cur = conn.cursor()
    cur.execute(f"SELECT COUNT(*) FROM vast_ads {where_clause}", params)
    total_rows = cur.fetchone()[0]
    cur.execute(f'''
        SELECT id, call_number, ad_id, creative_id, ssai_creative_id, title, duration, clickthrough, media_urls, adomain, creative_hash, created_at, wrapped_ad
        FROM vast_ads
        {where_clause}
        ORDER BY {sort} {order.upper()} LIMIT {per_page} OFFSET {offset}
    ''', params)
    rows = cur.fetchall()
    columns = ['id','call_number','ad_id','creative_id','ssai_creative_id','title','duration','clickthrough','media_urls','adomain','creative_hash','created_at','wrapped_ad']
    conn.close()
    # Parse media_urls JSON for each row
    parsed_rows = []
    for r in rows:
        r = list(r)
        try:
            r_media_urls = json.loads(r[columns.index('media_urls')]) if r[columns.index('media_urls')] else []
        except Exception:
            r_media_urls = []
        r.append(r_media_urls)  # Add as last element
        parsed_rows.append(r)
    # For ad comparison (select up to 2)
    compare_ids = request.args.getlist('compare')
    compare_ads = []
    compare_cols = []
    compare_table = []
    if compare_ids:
        conn = sqlite3.connect('vast_ads.db')
        cur = conn.cursor()
        qmarks = ','.join(['?']*len(compare_ids))
        cur.execute(f"SELECT * FROM vast_ads WHERE id IN ({qmarks})", compare_ids)
        compare_ads = cur.fetchall()
        compare_cols = [d[0] for d in cur.description]
        conn.close()
        # Build a zipped table for template: [(field, [ad1val, ad2val, ...]), ...]
        for i, field in enumerate(compare_cols):
            if field not in ['ad_xml','initial_metadata_json']:
                compare_table.append((field, [ad[i] for ad in compare_ads]))

    # Uniqueness summary for key fields and duplicate value sets
    def uniqueness_stats(rows, columns, field):
        idx = columns.index(field)
        vals = [r[idx] for r in rows]
        unique_vals = set(vals)
        duplicates = len(vals) - len(unique_vals)
        # Find duplicate values
        from collections import Counter
        val_counts = Counter(vals)
        dups = set([v for v, c in val_counts.items() if c > 1 and v])
        return {
            'field': field,
            'total': len(vals),
            'unique': len(unique_vals),
            'duplicates': duplicates,
            'dupset': dups,
        }

    summary_fields = ['ad_id', 'creative_id', 'ssai_creative_id', 'creative_hash']
    uniqueness = [uniqueness_stats(parsed_rows, columns, f) for f in summary_fields]
    # For easy lookup in table
    dup_lookup = {f['field']: f['dupset'] for f in uniqueness}

    # Bulk delete
    if request.method == 'POST' and request.form.get('action') == 'delete':
        ids_to_delete = request.form.getlist('delete_id')
        if ids_to_delete:
            conn = sqlite3.connect('vast_ads.db')
            cur = conn.cursor()
            qmarks = ','.join(['?']*len(ids_to_delete))
            cur.execute(f"DELETE FROM vast_ads WHERE id IN ({qmarks})", ids_to_delete)
            conn.commit()
            conn.close()
            return redirect(url_for('results'))

    # Precompute export CSV URL (Jinja2 does not support **request.args)
    from urllib.parse import urlencode
    args_dict = request.args.to_dict(flat=False)
    export_csv_url = url_for('export_csv')
    if request.args:
        export_csv_url += '?' + urlencode(request.args, doseq=True)

    # Precompute prev/next/sort URLs for pagination and sorting
    def build_url(**kwargs):
        # Merge current args with overrides
        merged = dict(request.args)
        merged.update(kwargs)
        # Remove keys with None values
        merged = {k: v for k, v in merged.items() if v is not None}
        return url_for('results') + ('?' + urlencode(merged, doseq=True) if merged else '')

    prev_url = build_url(page=page-1) if page > 1 else None
    next_url = build_url(page=page+1) if page * per_page < total_rows else None
    sort_urls = {}
    for col in columns[1:]:
        current_sort = request.args.get('sort')
        current_order = request.args.get('order', 'desc')
        if current_sort == col and current_order == 'desc':
            sort_urls[col] = build_url(sort=col, order='asc', page=page)
        else:
            sort_urls[col] = build_url(sort=col, order='desc', page=page)

    # List of all possible columns (excluding id)
    all_columns = columns[1:]
    import json as _json
    # Read column order/visibility from query param if present (for server-side rendering)
    vast_columns = request.args.get('vast_columns')
    if vast_columns:
        try:
            selected_columns = [c for c in vast_columns.split(',') if c in all_columns]
            if not selected_columns:
                selected_columns = all_columns[:]
        except Exception:
            selected_columns = all_columns[:]
    else:
        selected_columns = all_columns[:]
    all_columns_json = _json.dumps(all_columns)
    selected_columns_json = _json.dumps(selected_columns)
    export_db_url = url_for('export_db')
    return render_template_string('''
    <style>
      body {
        font-family: 'Inter', 'Segoe UI', Arial, sans-serif;
        background: linear-gradient(120deg, #f4f6fa 0%, #e8eefa 100%);
        margin: 0; padding: 0; color: var(--fg, #222);
      }
      .container {
        max-width: 1200px;
        margin: 40px auto;
        background: var(--panel, #fff);
        border-radius: 18px;
        box-shadow: 0 4px 32px #0002;
        padding: 36px 48px 48px 48px;
        position: relative;
      }
      h2 {
        margin-top: 0; font-size: 2.3em; letter-spacing: 1px;
        color: var(--h, #2a3b4c);
        background: linear-gradient(90deg, #2a7be4 10%, #4caf50 90%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
      }
      h4 { color: var(--h, #2a3b4c); margin-bottom: 10px; }
      .nav-links { margin-bottom: 18px; }
      .nav-links a {
        display: inline-block; margin-right: 18px; padding: 7px 18px;
        border-radius: 8px; background: #e8eefa; color: #2a3b4c;
        text-decoration: none; font-weight: 500; transition: background 0.2s, color 0.2s;
        box-shadow: 0 1px 4px #0001;
      }
      .nav-links a:hover { background: #2a3b4c; color: #fff; }
      .summary-table, .results-table {
        border-collapse: separate; border-spacing: 0 8px;
        width: 100%; background: transparent; margin-top: 18px;
      }
      .summary-table th, .summary-table td {
        padding: 10px 14px; border-bottom: 1px solid #e3e8f0; text-align: left;
      }
      .summary-table th { background: #e8eefa; color: #2a3b4c; font-weight: 600; }
      .results-table th {
        background: #e8eefa; color: #2a3b4c; font-weight: 600;
        cursor: pointer; position: sticky; top: 0; z-index: 2;
        border-radius: 8px 8px 0 0;
        box-shadow: 0 2px 8px #0001;
        border-right: 2px solid #e3e8f0;
      }
      .results-table th.sortable:hover { background: #2a3b4c; color: #fff; }
      .results-table tr {
        background: #fff;
        border-radius: 12px;
        box-shadow: 0 2px 12px #0001;
        transition: box-shadow 0.2s, transform 0.2s;
      }
      .results-table tr:hover {
        background: #f0f6ff;
        box-shadow: 0 6px 24px #2a7be420;
        /* Removed transform: scale to prevent row movement */
      }
      .results-table td {
        border-bottom: none;
        border-right: 2px solid #e3e8f0;
        padding: 12px 14px;
        font-size: 1.05em;
      }
      .results-table th:last-child, .results-table td:last-child {
        border-right: none;
      }
      .results-table thead { position: sticky; top: 0; z-index: 2; }
      .badge {
        display: inline-block;
        padding: 2px 10px;
        border-radius: 12px;
        font-size: 0.95em;
        font-weight: 600;
        color: #fff;
        margin-right: 4px;
      }
      .badge-wrapped { background: #2a7be4; }
      .badge-inline { background: #4caf50; }
      .badge-dup { background: #e53935; }
      .dup { background-color: #ffe0e0; }
      .filter-form { margin-bottom: 18px; display: flex; flex-wrap: wrap; gap: 10px; align-items: center; }
      .filter-form label { font-size: 0.95em; color: #2a3b4c; margin-right: 4px; }
      .filter-form input[type="text"] { padding: 7px; border-radius: 6px; border: 1px solid #d0d6e2; font-size: 1em; }
      .filter-form input[type="submit"], .filter-form a {
        background: linear-gradient(90deg, #2a7be4, #4caf50);
        color: #fff; border: none; border-radius: 8px; padding: 7px 18px;
        font-size: 1em; font-weight: 500; cursor: pointer; text-decoration: none; margin-left: 8px; transition: background 0.2s;
        box-shadow: 0 1px 4px #0001;
      }
      .filter-form input[type="submit"]:hover, .filter-form a:hover { background: #174a7c; }
      .delete-btn {
        background: linear-gradient(90deg, #e53935, #b71c1c);
        color: #fff; border: none; border-radius: 8px; padding: 7px 18px;
        font-size: 1em; font-weight: 500; cursor: pointer; margin-top: 12px; transition: background 0.2s;
        box-shadow: 0 1px 4px #0001;
      }
      .delete-btn:hover { background: #a31515; }
      .dark-toggle, .theme-select { float: right; margin-top: -40px; margin-left: 10px; }
      .compare-bar { margin: 18px 0; }
      .compare-bar button {
        background: linear-gradient(90deg, #2a7be4, #4caf50);
        color: #fff; border: none; border-radius: 8px; padding: 7px 18px;
        font-size: 1em; font-weight: 500; cursor: pointer; margin-right: 10px;
        box-shadow: 0 1px 4px #0001;
      }
      .compare-bar button:hover { background: #174a7c; }
      .tooltip { border-bottom: 1px dotted #888; cursor: help; }
      .spinner {
        display: none;
        position: fixed; left: 50%; top: 50%; z-index: 9999;
        width: 60px; height: 60px;
        border: 6px solid #e8eefa;
        border-top: 6px solid #2a7be4;
        border-radius: 50%;
        animation: spin 1s linear infinite;
      }
      @keyframes spin {
        0% { transform: rotate(0deg); }
        100% { transform: rotate(360deg); }
      }
      @media (prefers-color-scheme: dark) {
        :root { --bg: #181c20; --panel: #23272e; --fg: #e0e0e0; --h: #b8d0f6; }
      }
    </style>
    <div class="spinner" id="spinner"></div>
    <script>
      // Show spinner on navigation
      document.addEventListener('DOMContentLoaded', function() {
        Array.from(document.querySelectorAll('a,button,input[type="submit"]')).forEach(el => {
          el.addEventListener('click', function(e) {
            if (el.classList.contains('nav-links') || el.classList.contains('compare-bar') || el.classList.contains('delete-btn') || el.type === 'submit') {
              document.getElementById('spinner').style.display = 'block';
            }
          });
        });
      });
    </script>
    <script>
      // Theme selection
      function setTheme(theme) {
        document.body.classList.remove('theme-dark','theme-light','theme-contrast');
        document.body.classList.add('theme-' + theme);
        localStorage.setItem('vast_theme', theme);
      }
      function loadTheme() {
        var t = localStorage.getItem('vast_theme') || 'light';
        setTheme(t);
        document.getElementById('theme-select').value = t;
      }
      // Column customization
      const allColumns = {{ all_columns_json|safe }};
      // Always prefer vast_columns param from URL if present
      function getColumnsFromURL() {
        const params = new URLSearchParams(window.location.search);
        const v = params.get('vast_columns');
        if (v) {
          const arr = v.split(',').filter(c => allColumns.includes(c));
          if (arr.length) return arr;
        }
        return null;
      }
      function saveColumns(cols) {
        localStorage.setItem('vast_columns', JSON.stringify(cols));
      }
      function loadColumns() {
        const urlCols = getColumnsFromURL();
        if (urlCols) {
          saveColumns(urlCols); // keep localStorage in sync
          return urlCols;
        }
        let cols = localStorage.getItem('vast_columns');
        if (!cols) return allColumns.slice();
        try { return JSON.parse(cols); } catch { return allColumns.slice(); }
      }
      function updateColumnUI() {
        const cols = loadColumns();
        // Update checkboxes
        allColumns.forEach(col => {
          const cb = document.getElementById('col_cb_' + col);
          if (cb) cb.checked = cols.includes(col);
        });
        // Update order
        const orderBox = document.getElementById('col_order');
        if (!orderBox) return;
        orderBox.innerHTML = '';
        cols.forEach(col => {
          const li = document.createElement('li');
          li.textContent = col;
          li.draggable = true;
          li.id = 'order_' + col;
          li.ondragstart = e => { e.dataTransfer.setData('text/plain', col); };
          li.ondragover = e => { e.preventDefault(); };
          li.ondrop = e => {
            e.preventDefault();
            const dragged = e.dataTransfer.getData('text/plain');
            const idx = cols.indexOf(dragged);
            const tgtIdx = cols.indexOf(col);
            if(idx>-1 && tgtIdx>-1 && idx!=tgtIdx) {
              cols.splice(idx,1);
              cols.splice(tgtIdx,0,dragged);
              saveColumns(cols);
              // Always update server after drop
              updateServerColumns(cols);
            }
          };
          orderBox.appendChild(li);
        });
      }
      function updateServerColumns(cols) {
        // Update the URL with the new vast_columns param and reload
        const params = new URLSearchParams(window.location.search);
        params.set('vast_columns', cols.join(','));
        window.location.search = params.toString();
      }
      function onColChange() {
        const cols = allColumns.filter(col => {
          const cb = document.getElementById('col_cb_' + col);
          return cb && cb.checked;
        });
        saveColumns(cols);
        updateServerColumns(cols);
      }
      window.addEventListener('DOMContentLoaded', function() {
        loadTheme();
        updateColumnUI();
      });
      function compareAds() {
        const checked = Array.from(document.querySelectorAll('input[name="compare_id"]:checked')).map(cb=>cb.value);
        if(checked.length<2) { alert('Select at least 2 ads to compare.'); return; }
        const params = new URLSearchParams(window.location.search);
        params.delete('compare');
        checked.forEach(id=>params.append('compare',id));
        window.location.search = params.toString();
      }
    </script>
    <div class="container">
      <div class="header-bar" style="display:flex;align-items:center;justify-content:space-between;gap:24px;margin-bottom:18px;">
        <div style="display:flex;align-items:center;gap:12px;">
          <a href="/" class="nav-btn">Parse One</a>
          <a href="/multi" class="nav-btn">Parse Multiple</a>
        </div>
        <div class="theme-switcher" style="display:flex;align-items:center;gap:6px;">
          <label for="theme-select" style="font-size:1em;font-weight:500;">Theme:</label>
          <select id="theme-select" class="theme-select" onchange="setTheme(this.value)">
            <option value="light">Light</option>
            <option value="dark">Dark</option>
            <option value="contrast">High Contrast</option>
          </select>
        </div>
      </div>
      <h2 style="margin-bottom:18px;">Parsed Ads</h2>
      <form method="get" class="filter-form" style="margin-bottom:0;">
        <details style="margin-bottom:16px;">
          <summary style="font-weight:500;cursor:pointer;">Customize Columns</summary>
          <div style="margin:10px 0 0 0;">
            <div style="display:flex;gap:24px;align-items:flex-start;">
              <div>
                <strong>Show Columns:</strong><br>
                {% for col in all_columns %}
                  <label><input type="checkbox" id="col_cb_{{col}}" onchange="onColChange()" {% if col in selected_columns %}checked{% endif %}> {{col}}</label><br>
                {% endfor %}
              </div>
              <div>
                <strong>Order:</strong>
                <ul id="col_order" style="list-style:none;padding:0;margin:0;">
                  {% for col in selected_columns %}
                    <li draggable="true" id="order_{{col}}">{{col}}</li>
                  {% endfor %}
                </ul>
                <small>Drag to reorder</small>
              </div>
            </div>
          </div>
        </details>
        <div class="filter-grid" style="display:grid;grid-template-columns:repeat(4,1fr);gap:10px 18px;margin-bottom:12px;align-items:end;">
          <div style="grid-column:span 4;">
            <label style="font-weight:500;">Global Search:</label>
            <input type="text" name="q" value="{{ request.args.get('q','') }}" placeholder="Any field" style="width:60%;max-width:340px;">
          </div>
          {% for f in ['ad_id','creative_id','ssai_creative_id','title','duration','clickthrough','adomain','creative_hash'] %}
            <div>
              <label style="font-weight:500;">{{ f|replace('_',' ')|title }}:</label>
              <input type="text" name="{{f}}" value="{{ request.args.get(f,'') }}" style="width:100%;">
            </div>
          {% endfor %}
        </div>
        <div class="action-bar" style="display:flex;gap:12px;align-items:center;margin-bottom:0;">
          <button type="submit" class="action-btn filter-btn">Filter</button>
          <a href="{{ export_csv_url }}" class="action-btn export-btn">Export CSV</a>
          <a href="{{ export_db_url }}" class="action-btn export-btn">Export Full DB</a>
          <button type="button" onclick="compareAds()" class="action-btn compare-btn">Compare Selected</button>
          <span style="color:#888;font-size:0.98em;">(Select 2 ads to compare side-by-side)</span>
        </div>
      </form>
      <h4 style="margin-top:28px;">Identifier Uniqueness Summary</h4>
      <table class="summary-table">
        <tr><th>Field</th><th>Total</th><th>Unique</th><th>Duplicates</th></tr>
        {% for stat in uniqueness %}
          <tr>
            <td><span class="tooltip" title="{{ stat.field }}">{{ stat.field }}</span></td>
            <td>{{ stat.total }}</td>
            <td>{{ stat.unique }}</td>
            <td>{% if stat.duplicates > 0 %}<span style="color:red">{{ stat.duplicates }}</span>{% else %}0{% endif %}</td>
          </tr>
        {% endfor %}
      </table>
      <form method="post">
      <input type="hidden" name="action" value="delete">
      <table class="results-table" style="margin-top:18px;">
        <thead>
        <tr id="thead-row">
          <th style="width:32px;"></th>
          <th style="width:32px;"></th>
          {% for col in selected_columns %}
            <th class="sortable" title="Sort by {{ col }}" data-col="{{col}}">
              <a href="{{ sort_urls[col] }}">{{ col|replace('_', ' ')|title }}</a>
            </th>
          {% endfor %}
          <th>Details</th>
        </tr>
        </thead>
        <tbody>
        {% for r in parsed_rows %}
        <tr>
          <td><input type="checkbox" name="delete_id" value="{{ r[0] }}"></td>
          <td><input type="checkbox" name="compare_id" value="{{ r[0] }}"></td>
          {% for col in selected_columns %}
            {% set idx = columns.index(col) %}
            {% if col == 'media_urls' %}
              <td data-col="{{col}}">
                {% set urls = r[-1] %}
                {% if urls %}
                  {% for url in urls %}
                    <a href="{{ url }}" target="_blank">{{ url }}</a><br>
                  {% endfor %}
                {% endif %}
              </td>
            {% elif col == 'wrapped_ad' %}
              <td data-col="{{col}}">
                {% if r[idx] %}
                  <span class="badge badge-wrapped">Wrapped</span>
                {% else %}
                  <span class="badge badge-inline">Inline</span>
                {% endif %}
              </td>
            {% elif col in dup_lookup and r[idx] in dup_lookup[col] and r[idx] %}
              <td class="dup" data-col="{{col}}" title="Duplicate value">
                <span class="badge badge-dup">Dup</span> {{ r[idx] }}
              </td>
            {% else %}
              <td data-col="{{col}}" title="{{ col }}">{{ r[idx] }}</td>
            {% endif %}
          {% endfor %}
          <td><a href="{{ url_for('ad_details', ad_id=r[2]) }}">View</a></td>
        </tr>
        {% endfor %}
        </tbody>
      </table>
      <input class="delete-btn" type="submit" value="Delete Selected" onclick="return confirm('Delete selected ads?')">
      </form>
      <div style="margin:18px 0;">
        {% if prev_url %}<a href="{{ prev_url }}">&larr; Prev</a>{% endif %}
        <span style="margin:0 12px;">Page {{ page }} of {{ (total_rows // per_page) + (1 if total_rows % per_page else 0) }}</span>
        {% if next_url %}<a href="{{ next_url }}">Next &rarr;</a>{% endif %}
      </div>
      {% if compare_ads and compare_ads|length >= 2 %}
        <div style="margin:32px 0; background:#fafdff; border-radius:8px; padding:24px; box-shadow:0 2px 8px #0001;">
          <h4>Ad Comparison</h4>
          <table style="width:100%; border-collapse:collapse;">
            <tr>
              <th>Field</th>
              {% for ad in compare_ads %}<th>Ad {{ ad[2] }}</th>{% endfor %}
            </tr>
            {% for field, values in compare_table %}
              <tr>
                <td><span class="tooltip" title="{{ field }}">{{ field }}</span></td>
                {% for val in values %}
                  <td>{{ val }}</td>
                {% endfor %}
              </tr>
            {% endfor %}
          </table>
        </div>
      {% endif %}
    </div>
    ''', parsed_rows=parsed_rows, columns=columns, uniqueness=uniqueness, page=page, per_page=per_page, total_rows=total_rows, compare_ads=compare_ads if 'compare_ads' in locals() else [], compare_cols=compare_cols if 'compare_ads' in locals() else [], compare_table=compare_table if 'compare_ads' in locals() else [], export_csv_url=export_csv_url, export_db_url=export_db_url, prev_url=prev_url, next_url=next_url, sort_urls=sort_urls, all_columns=all_columns, selected_columns=selected_columns)

# Export CSV endpoint
@app.route('/export_csv')
def export_csv():
    # If any filters are present, export filtered results; otherwise, export the entire table
    sort = request.args.get('sort', 'id')
    order = request.args.get('order', 'desc')
    allowed_sorts = ['id', 'call_number', 'ad_id', 'creative_id', 'ssai_creative_id', 'title', 'duration', 'clickthrough', 'media_urls', 'adomain', 'creative_hash', 'created_at']
    if sort not in allowed_sorts:
        sort = 'id'
    if order not in ['asc', 'desc']:
        order = 'desc'
    # Check for any filter params
    filter_fields = ['ad_id','creative_id','ssai_creative_id','title','duration','clickthrough','adomain','creative_hash']
    filters = {f: request.args.get(f, '').strip() for f in filter_fields}
    filters = {k: v for k, v in filters.items() if v}
    global_search = request.args.get('q', '').strip()
    where = []
    params = []
    if filters:
        for k, v in filters.items():
            where.append(f"{k} LIKE ?")
            params.append(f"%{v}%")
    if global_search:
        search_fields = ['ad_id','creative_id','ssai_creative_id','title','duration','clickthrough','media_urls','adomain','creative_hash']
        where.append('(' + ' OR '.join([f"{f} LIKE ?" for f in search_fields]) + ')')
        params.extend([f"%{global_search}%"]*len(search_fields))
    where_clause = f"WHERE {' AND '.join(where)}" if where else ''
    conn = sqlite3.connect('vast_ads.db')
    cur = conn.cursor()
    cur.execute(f'''
        SELECT call_number, ad_id, creative_id, ssai_creative_id, title, duration, clickthrough, media_urls, adomain, creative_hash, created_at
        FROM vast_ads
        {where_clause}
        ORDER BY {sort} {order.upper()}
    ''', params)
    rows = cur.fetchall()
    conn.close()
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(['call_number','ad_id','creative_id','ssai_creative_id','title','duration','clickthrough','media_urls','adomain','creative_hash','created_at'])
    for row in rows:
        # media_urls is JSON, flatten for CSV
        row = list(row)
        row[7] = ','.join(json.loads(row[7])) if row[7] else ''
        writer.writerow(row)
    output.seek(0)
    return send_file(io.BytesIO(output.getvalue().encode()), mimetype='text/csv', as_attachment=True, download_name='vast_ads.csv')

# Ad details view with raw JSON and raw XML
@app.route('/ad/<ad_id>')
def ad_details(ad_id):
    conn = sqlite3.connect('vast_ads.db')
    cur = conn.cursor()
    cur.execute('''
        SELECT call_number, ad_id, creative_id, ssai_creative_id, title, duration, clickthrough, media_urls, channel_name, adomain, creative_hash, created_at, ad_xml, wrapped_ad, initial_metadata_json
        FROM vast_ads WHERE ad_id = ? LIMIT 1
    ''', (ad_id,))
    row = cur.fetchone()
    columns = ['call_number','ad_id','creative_id','ssai_creative_id','title','duration','clickthrough','media_urls','channel_name','adomain','creative_hash','created_at','ad_xml','wrapped_ad','initial_metadata_json']
    conn.close()
    if not row:
        return "Ad not found", 404
    media_urls = json.loads(row[7]) if row[7] else []
    # Build ad dict for JSON view
    ad_dict = {col: row[i] for i, col in enumerate(columns[:-2])}
    ad_dict['media_urls'] = media_urls
    import json as _json
    raw_json = _json.dumps(ad_dict, indent=2)
    show_json = request.args.get('show_json') == '1'
    show_xml = request.args.get('show_xml') == '1'
    show_initial = request.args.get('show_initial') == '1'

    ad_xml = row[12] if show_xml else None
    xml_error = None
    initial_metadata_json = row[-1]
    initial_metadata_pretty = None
    if initial_metadata_json:
        try:
            initial_metadata_pretty = _json.dumps(_json.loads(initial_metadata_json), indent=2)
        except Exception:
            initial_metadata_pretty = initial_metadata_json

    return render_template_string('''
      <style>
      .nav-btn {
        background: #e8eefa;
        color: #2a3b4c;
        border: none;
        border-radius: 6px;
        padding: 7px 18px;
        font-weight: 500;
        text-decoration: none;
        margin-right: 6px;
        transition: background 0.2s, color 0.2s;
      }
      .nav-btn:hover { background: #2a3b4c; color: #fff; }
      .action-bar .action-btn {
        background: #2a7be4;
        color: #fff;
        border: none;
        border-radius: 6px;
        padding: 8px 20px;
        font-size: 1.08em;
        font-weight: 500;
        cursor: pointer;
        text-decoration: none;
        transition: background 0.2s;
        box-shadow: 0 1px 4px #0001;
      }
      .action-bar .action-btn.filter-btn { background: #4caf50; }
      .action-bar .action-btn.export-btn { background: #2a7be4; }
      .action-bar .action-btn.compare-btn { background: #388e3c; }
      .action-bar .action-btn:hover { background: #174a7c; color: #fff; }
      .theme-switcher select.theme-select {
        border-radius: 6px;
        padding: 4px 10px;
        font-size: 1em;
        border: 1px solid #d0d6e2;
        background: #fff;
        color: #2a3b4c;
        margin-left: 4px;
      }
      body { font-family: 'Segoe UI', Arial, sans-serif; background: #f4f6fa; margin: 0; padding: 0; }
      .container { max-width: 900px; margin: 40px auto; background: #fff; border-radius: 12px; box-shadow: 0 2px 12px #0001; padding: 32px 40px 40px 40px; }
      h2 { margin-top: 0; font-size: 2.2em; letter-spacing: 1px; color: #2a3b4c; }
      .toggle-bar { margin-bottom: 18px; }
      .toggle-bar a { display: inline-block; margin-right: 18px; padding: 7px 18px; border-radius: 6px; background: #e8eefa; color: #2a3b4c; text-decoration: none; font-weight: 500; transition: background 0.2s; }
      .toggle-bar a.active, .toggle-bar a:hover { background: #2a3b4c; color: #fff; }
      pre { background: #f8f8f8; border: 1px solid #d0d6e2; border-radius: 6px; padding: 16px; font-size: 1em; overflow-x: auto; }
      .section { margin-bottom: 28px; }
      table { border-collapse: collapse; width: 100%; background: #fafdff; border-radius: 8px; overflow: hidden; margin-top: 18px; }
      th, td { padding: 10px 14px; border-bottom: 1px solid #e3e8f0; text-align: left; }
      th { background: #e8eefa; color: #2a3b4c; font-weight: 600; }
      tr:last-child td { border-bottom: none; }
      .wrapped { color: #2a7be4; font-weight: bold; }
      .inline { color: #4caf50; font-weight: bold; }
      .back-link { display: inline-block; margin-top: 24px; color: #2a7be4; text-decoration: none; font-weight: 500; }
      .back-link:hover { text-decoration: underline; }
    </style>
    <div class="container">
      <h2>Ad Details</h2>
      <div class="toggle-bar">
        <a href="{{ url_for('ad_details', ad_id=row[1], show_json='1' if not show_json else None, show_xml=request.args.get('show_xml'), show_initial=request.args.get('show_initial')) }}" class="{{ 'active' if show_json else '' }}">{{ 'Show Raw JSON' if not show_json else 'Hide Raw JSON' }}</a>
        <a href="{{ url_for('ad_details', ad_id=row[1], show_xml='1' if not show_xml else None, show_json=request.args.get('show_json'), show_initial=request.args.get('show_initial')) }}" class="{{ 'active' if show_xml else '' }}">{{ 'Show Raw XML' if not show_xml else 'Hide Raw XML' }}</a>
        <a href="{{ url_for('ad_details', ad_id=row[1], show_initial='1' if not show_initial else None, show_json=request.args.get('show_json'), show_xml=request.args.get('show_xml')) }}" class="{{ 'active' if show_initial else '' }}">{{ 'Show Initial Metadata' if not show_initial else 'Hide Initial Metadata' }}</a>
      </div>

      {% if show_json %}
        <div class="section">
          <h4>Raw JSON</h4>
          <pre>{{ raw_json }}</pre>
        </div>
      {% endif %}

      {% if show_xml %}
        <div class="section">
          <h4>Raw XML</h4>
          {% if ad_xml %}
            <pre>{{ ad_xml }}</pre>
          {% elif xml_error %}
            <div style="color:red;">{{ xml_error }}</div>
          {% endif %}
        </div>
      {% endif %}

      {% if show_initial and initial_metadata_pretty %}
        <div class="section">
          <h4>Initial Wrapper Metadata</h4>
          <pre>{{ initial_metadata_pretty }}</pre>
        </div>
      {% endif %}

      <div class="section">
        <h4>Summary</h4>
        <table>
          {% for col in columns[:-3] %}
            <tr>
              <th>{{ col }}</th>
              <td>
                {% if col == 'media_urls' %}
                  {% for url in media_urls %}
                    <a href="{{ url }}" target="_blank">{{ url }}</a><br>
                  {% endfor %}
                {% else %}
                  {{ row[columns.index(col)] }}
                {% endif %}
              </td>
            </tr>
          {% endfor %}
          <tr>
            <th>wrapped_ad</th>
            <td>{% if row[-2] %}<span class="wrapped">Wrapped</span>{% else %}<span class="inline">Inline</span>{% endif %}</td>
          </tr>
        </table>
      </div>
      <a class="back-link" href="{{ url_for('results') }}">&larr; Back to Results</a>
    </div>
    ''', row=row, columns=columns, media_urls=media_urls, raw_json=raw_json, show_json=show_json, ad_xml=ad_xml, show_xml=show_xml, xml_error=xml_error, show_initial=show_initial, initial_metadata_pretty=initial_metadata_pretty)

if __name__ == '__main__':
    app.run(debug=True)
