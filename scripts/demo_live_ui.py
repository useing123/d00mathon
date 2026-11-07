"""Playwright walk-through of docs/DEMO_LIVE.md (add 09 -> save -> re-evaluate one call -> replace 05 -> restore).

Run against a NON-production server (it edits checklists/freedom_call_v1.yaml and the group results):
    /usr/bin/python3 scripts/demo_live_ui.py --url http://127.0.0.1:8095 --out docs/img
Prints the job meta (LLM calls / cost / time) of each re-evaluation and the console errors (must be 0).
"""
import argparse
import json
import sys
import time
from pathlib import Path

from playwright.sync_api import sync_playwright

CALL = "p4-ru-001"
NAME09 = "Предупреждение о записи разговора"
RUB09 = ("YES — оператор или IVR сообщил клиенту, что разговор записывается.\n"
         "NO — о записи разговора не сообщено.\nNA — звонок прервался до разговора с оператором.")
ADD05 = "\nДополнительно: если клиент недоволен, оператор признаёт неудобство своими словами, а не только шаблонной фразой."


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8095")
    ap.add_argument("--group", default="DEV")
    ap.add_argument("--out", default="docs/img")
    ap.add_argument("--orig", required=True, help="original freedom_call_v1.yaml to paste back in step 8")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    errors, report = [], {}
    with sync_playwright() as p:
        b = p.chromium.launch()
        pg = b.new_page(viewport={"width": 1440, "height": 900})
        pg.on("console", lambda m: m.type == "error" and errors.append(m.text))
        pg.on("pageerror", lambda e: errors.append(str(e)))
        shot = lambda n: pg.screenshot(path=str(out / f"demo_{n}.png"))  # noqa: E731

        def checklist_tab():
            pg.click("#tabs a[data-v=checklists]")
            pg.wait_for_selector("#clSel option", state="attached")
            pg.select_option("#clSel", "freedom_call_v1")
            pg.wait_for_timeout(300)

        def save():
            pg.click("#btnClSave")
            pg.wait_for_function("document.querySelector('#clMsg').textContent.startsWith('Сохранено')")

        def reeval(tag):
            pg.click("#tabs a[data-v=runs]")
            pg.click(f"tr:has-text('{CALL}')")
            pg.click("button:has-text('Переоценить')")
            pg.wait_for_selector("#askM:not(.hide)")
            shot(f"{tag}_confirm")
            t = time.time()
            pg.click("#askOk")
            pg.wait_for_selector("#pclose:not(.hide)", timeout=180000)
            report[tag] = {"ui_s": round(time.time() - t, 1), "status": pg.inner_text("#pstat"), "meta": pg.inner_text("#pmeta")}
            shot(f"{tag}_done")
            pg.click("#pclose")

        pg.goto(a.url)
        pg.wait_for_selector("#gsel option", state="attached")
        pg.select_option("#gsel", a.group)
        pg.wait_for_selector(f"tr:has-text('{CALL}')")
        pg.click(f"tr:has-text('{CALL}')")
        pg.wait_for_timeout(500)
        shot("01_dialog_before")

        # step 2-3: add criterion 09 in the form, save
        checklist_tab()
        pg.click("#btnAddCrit")
        card = pg.locator(".ccard").last
        card.locator("label.f:has-text('Название') input").fill(NAME09)
        card.locator("label.f:has-text('Код') input").fill("recording_notice")
        card.locator("label.f:has-text('Вес') input").fill("5")
        card.locator("label.f:has-text('Вес') input").press("Tab")
        card = pg.locator(".ccard").last
        card.locator("label.ck:has-text('Допускает н/п') input").check()
        card.locator("label.f:has-text('Рубрика') textarea").fill(RUB09)
        card.locator("label.f:has-text('Рубрика') textarea").press("Tab")
        card.scroll_into_view_if_needed()
        shot("02_add_criterion")
        save()
        pg.evaluate("window.scrollTo(0,0)")
        shot("03_saved")

        # step 4-5: re-evaluate the one call (cache -> only 09 goes to the model)
        reeval("04_reeval09")
        card9 = pg.locator("#crits > *:has-text('Предупреждение о записи')").first
        card9.scroll_into_view_if_needed()
        card9.click()
        pg.wait_for_timeout(400)
        shot("05_dialog_09")
        report["dialog_total_text"] = pg.inner_text("#v-dialog")[:300]

        # step 6: export menu + v1 file
        pg.click("#btnMore")
        shot("06_export_menu")
        v1 = pg.evaluate(f"fetch('/api/export/{a.group}/results.json').then(r=>r.json())")
        rec = next(r for r in v1["results"] if r["conversation_id"] == CALL)
        report["v1_codes"] = [c["criterion_id"] for c in rec["criteria"]]
        report["v1_score"] = rec.get("score")
        pg.keyboard.press("Escape")
        pg.click("body", position={"x": 5, "y": 300})

        # step 7: replace 05 (description/rubric), save, re-evaluate -> only 05
        checklist_tab()
        c5 = pg.locator(".ccard", has=pg.locator("label.f:has-text('Название') input[value='Эмпатия']")).first
        ta = c5.locator("label.f:has-text('Рубрика') textarea")
        ta.fill(ta.input_value().rstrip() + ADD05)
        ta.press("Tab")
        c5.scroll_into_view_if_needed()
        shot("07_replace_05")
        save()
        reeval("08_reeval05")

        # step 9: restore the original YAML via the YAML editor
        checklist_tab()
        pg.click("#btnYaml")
        pg.wait_for_function("document.querySelector('#yaml').value.length>0")
        pg.fill("#yaml", Path(a.orig).read_text(encoding="utf-8"))
        save()
        shot("09_restored")
        b.close()
    report["console_errors"] = errors
    print(json.dumps(report, ensure_ascii=False, indent=1))
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
