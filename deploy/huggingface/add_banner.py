"""Add a small, script-free notice to the built UI (the page's CSP allows inline styles, not scripts)."""

import sys
from pathlib import Path

index = Path(sys.argv[1])
hours = sys.argv[2] if len(sys.argv) > 2 else "6"
notice = (
    '<div role="note" style="position:fixed;left:8px;bottom:8px;z-index:9999;max-width:200px;padding:6px 9px;'
    'border-radius:8px;background:rgba(15,23,42,.92);border:1px solid #334155;color:#cbd5e1;'
    'font:11px/1.35 system-ui,sans-serif;pointer-events:none">'
    '<b style="color:#fbbf24">Public demo</b> &middot; models are simulated (real models run locally with Ollama). '
    f'Shared space, reset every {hours} h &ndash; do not upload private files.</div>'
)
html = index.read_text(encoding="utf-8")
if 'role="note"' not in html:
    html = html.replace("</body>", notice + "</body>", 1)
    index.write_text(html, encoding="utf-8")
