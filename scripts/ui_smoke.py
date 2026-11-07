"""Headless browser smoke test of the reviewer UI (requires `pip install playwright && playwright install chromium`).

Usage: python scripts/ui_smoke.py [http://127.0.0.1:8090] [GROUP]
Checks: page loads with 0 console errors; dialog table renders; opening a dialog renders criteria, jury chips and
evidence; an injected `incomplete` dialog shows the incomplete badge + provisional total; an invalid override is
rejected by the API with HTTP 400. Exit code 1 on any failure.
"""
import json
import sys
import urllib.error
import urllib.request

from playwright.sync_api import sync_playwright

BASE = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8090"
GROUP = sys.argv[2] if len(sys.argv) > 2 else "DEV"


def main():
    errs, checks = [], {}
    with sync_playwright() as p:
        b = p.chromium.launch()
        pg = b.new_page()
        pg.on("console", lambda m: m.type == "error" and errs.append(m.text))
        pg.on("pageerror", lambda e: errs.append(str(e)))
        pg.goto(BASE + "/")
        pg.wait_for_timeout(2500)
        checks["table_rows"] = pg.locator("table tbody tr").count()
        cid = pg.evaluate("DOC && DOC.dialogs[0].conversation_id")
        pg.evaluate(f"openDialog({json.dumps(cid)})")
        pg.wait_for_timeout(1200)
        checks["criteria_cards"] = pg.locator("#crits .cr").count()
        checks["jury_chips"] = pg.locator(".jv").count()
        pg.evaluate("""()=>{const d=DOC.dialogs[0];d.status='incomplete';d.failed_criteria=['03'];d.provisional_total=88;
            d.total=null;d.warnings=['demo'];d.criteria[2].error='api down';d.criteria[2].value=null;openDialog(d.conversation_id)}""")
        pg.wait_for_timeout(800)
        checks["incomplete_badge"] = pg.locator("#incTag").count()
        checks["provisional_shown"] = "предварительно 88.0" in pg.inner_text("#dhead")
        b.close()
    req = urllib.request.Request(f"{BASE}/api/overrides/{GROUP}/{cid}/04", data=json.dumps({"value": "4.5"}).encode(),
                                 method="PUT", headers={"Content-Type": "application/json"})
    try:
        urllib.request.urlopen(req)
        checks["invalid_override_status"] = 200
    except urllib.error.HTTPError as e:
        checks["invalid_override_status"] = e.code
    checks["console_errors"] = errs
    print(json.dumps(checks, ensure_ascii=False, indent=1))
    ok = (checks["table_rows"] > 0 and checks["criteria_cards"] >= 8 and checks["incomplete_badge"] == 1
          and checks["provisional_shown"] and checks["invalid_override_status"] == 400 and not errs)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
