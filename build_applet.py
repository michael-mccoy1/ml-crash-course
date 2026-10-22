"""Inject data into applet/applet_template.html -> docs/index.html (served by GitHub Pages)."""
import os
here = os.path.dirname(os.path.abspath(__file__))
tpl = open(os.path.join(here, "applet", "applet_template.html"), encoding="utf-8").read()
data = open(os.path.join(here, "results", "applet_data.json"), encoding="utf-8").read()
html = "<!doctype html>\n<html lang=\"en\"><head><meta charset=\"utf-8\"><meta name=\"viewport\" content=\"width=device-width,initial-scale=1\"></head><body>\n" + tpl.replace("/*DATA*/", data) + "\n</body></html>\n"
os.makedirs(os.path.join(here, "docs"), exist_ok=True)
open(os.path.join(here, "docs", "index.html"), "w", encoding="utf-8").write(html)
open(os.path.join(here, "docs", ".nojekyll"), "w").close()
print("wrote docs/index.html", len(html) // 1024, "KB")
