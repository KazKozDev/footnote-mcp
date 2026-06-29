"""Browser automation tools via Playwright (headless Chromium)."""

import asyncio
import re
from pathlib import Path

from playwright.async_api import async_playwright


def _ocr_png_bytes(data: bytes) -> str:
    """OCR a PNG screenshot. Returns extracted text or a short note if OCR is unavailable."""
    try:
        import io

        import pytesseract
        from PIL import Image

        return pytesseract.image_to_string(Image.open(io.BytesIO(data))).strip()
    except Exception as exc:
        return f"[ocr unavailable: {exc}]"


INTERACTIVE_SNAPSHOT_JS = """
() => {
  const selectors = [
    'a[href]',
    'button',
    'input',
    'textarea',
    'select',
    '[contenteditable="true"]',
    '[role="button"]',
    '[role="link"]',
    '[role="textbox"]',
    '[role="searchbox"]',
    '[role="combobox"]',
    '[role="checkbox"]',
    '[role="radio"]',
    '[role="menuitem"]',
    '[role="option"]',
    '[role="tab"]',
    '[role="switch"]'
  ].join(',');

  document.querySelectorAll('[data-footnote-ref]').forEach((el) => {
    el.removeAttribute('data-footnote-ref');
  });

  const roleFor = (el) => {
    const explicit = el.getAttribute('role');
    if (explicit) return explicit;
    const tag = el.tagName.toLowerCase();
    const type = (el.getAttribute('type') || '').toLowerCase();
    if (tag === 'a') return 'link';
    if (tag === 'button' || type === 'button' || type === 'submit') return 'button';
    if (type === 'checkbox') return 'checkbox';
    if (type === 'radio') return 'radio';
    if (tag === 'select') return 'combobox';
    if (tag === 'textarea') return 'textbox';
    if (tag === 'input' && (type === 'search' || el.getAttribute('name') === 'search')) return 'searchbox';
    if (tag === 'input') return 'textbox';
    return 'button';
  };

  const nameFor = (el) => {
    const id = el.id;
    const label = id ? document.querySelector(`label[for="${CSS.escape(id)}"]`) : null;
    return (
      el.getAttribute('aria-label') ||
      (label && label.innerText) ||
      el.getAttribute('title') ||
      el.getAttribute('placeholder') ||
      el.innerText ||
      el.value ||
      el.getAttribute('alt') ||
      el.getAttribute('href') ||
      ''
    ).trim().replace(/\\s+/g, ' ').slice(0, 200);
  };

  const visible = (el) => {
    const style = window.getComputedStyle(el);
    const rect = el.getBoundingClientRect();
    return style.display !== 'none' &&
      style.visibility !== 'hidden' &&
      rect.width > 0 &&
      rect.height > 0 &&
      !el.disabled &&
      el.getAttribute('aria-hidden') !== 'true';
  };

  return Array.from(document.querySelectorAll(selectors))
    .filter(visible)
    .slice(0, 200)
    .map((el, index) => {
      const ref = `@e${index + 1}`;
      el.setAttribute('data-footnote-ref', ref);
      return {
        ref,
        role: roleFor(el),
        name: nameFor(el),
        tag: el.tagName.toLowerCase(),
        type: (el.getAttribute('type') || '').toLowerCase()
      };
    });
}
"""


class WebBrowser:
    """ponytail: one class, one browser instance, lazy init."""

    def __init__(self, headed: bool = False):
        self.headed = headed
        self._playwright = None
        self._browser = None
        self._page = None
        self._snapshot_elements: list[dict] = []

    async def _ensure_browser(self):
        if self._browser is not None:
            return
        self._playwright = await async_playwright().start()
        self._browser = await self._playwright.chromium.launch(headless=not self.headed)
        self._page = await self._browser.new_page()

    async def navigate(self, url: str) -> dict:
        await self._ensure_browser()
        await self._page.goto(url, wait_until="domcontentloaded", timeout=30000)
        title = await self._page.title()
        return {"url": self._page.url, "title": title}

    async def snapshot(self) -> dict:
        """Return visible interactive elements with stable refs like @e1, @e2, ..."""
        await self._ensure_browser()
        elements = await self._page.evaluate(INTERACTIVE_SNAPSHOT_JS)
        self._snapshot_elements = elements

        url = self._page.url
        title = await self._page.title()
        visible_text = await self._page.evaluate("() => document.body?.innerText?.slice(0, 3000) || ''")

        return {
            "url": url,
            "title": title,
            "elements": elements,
            "visible_text": visible_text,
            "element_count": len(elements),
        }

    async def _find_element(self, ref: str) -> dict | None:
        """Find snapshot metadata by ref."""
        for el in self._snapshot_elements:
            if el["ref"] == ref:
                return el
        return None

    async def _locator_for_ref(self, ref: str):
        if not re.fullmatch(r"@e\d+", ref):
            return None
        locator = self._page.locator(f'[data-footnote-ref="{ref}"]').first
        if await locator.count() == 0:
            return None
        return locator

    async def click(self, ref: str) -> dict:
        await self._ensure_browser()
        info = await self._find_element(ref)
        if not info:
            return {"ok": False, "error": f"Ref {ref} not found in snapshot. Take a new snapshot first."}

        locator = await self._locator_for_ref(ref)
        if locator is None:
            return {"ok": False, "error": f"Ref {ref} is stale. Take a new snapshot first."}
        try:
            await locator.click(timeout=5000)
            await asyncio.sleep(0.5)
            return {"ok": True, "url": self._page.url, "title": await self._page.title()}
        except Exception as e:
            return {"ok": False, "error": str(e)}

    async def type(self, ref: str, text: str, submit: bool = False) -> dict:
        await self._ensure_browser()
        info = await self._find_element(ref)
        if not info:
            return {"ok": False, "error": f"Ref {ref} not found. Take a new snapshot first."}

        locator = await self._locator_for_ref(ref)
        if locator is None:
            return {"ok": False, "error": f"Ref {ref} is stale. Take a new snapshot first."}
        try:
            await locator.fill("")
            await locator.type(text)
            if submit:
                await locator.press("Enter")
            return {"ok": True, "url": self._page.url, "title": await self._page.title()}
        except Exception as e:
            return {"ok": False, "error": str(e)}

    async def extract(self, refs: str) -> dict:
        """Extract text from specific refs or whole page. refs: comma-separated @eN, or 'visible', or 'all'."""
        await self._ensure_browser()

        if refs == "visible":
            text = await self._page.evaluate("() => document.body?.innerText || ''")
            return {"text": text[:8000]}

        if refs == "all":
            html = await self._page.evaluate("() => document.documentElement?.outerHTML?.slice(0, 50000) || ''")
            return {"html": html}

        # specific refs
        ref_list = [r.strip() for r in refs.split(",")]
        results = {}
        for ref in ref_list:
            info = await self._find_element(ref)
            if info:
                locator = await self._locator_for_ref(ref)
                if locator is None:
                    results[ref] = ""
                    continue
                try:
                    el_text = await locator.text_content(timeout=3000)
                    results[ref] = (el_text or "").strip()
                except Exception:
                    results[ref] = ""
            else:
                results[ref] = ""
        return {"elements": results}

    async def extract_tables(self, max_tables: int = 8, max_rows: int = 100) -> dict:
        """Extract visible HTML tables from the current browser page."""
        await self._ensure_browser()
        tables = await self._page.evaluate(
            """
            ({maxTables, maxRows}) => {
              const result = [];
              const visible = (el) => {
                const style = window.getComputedStyle(el);
                const rect = el.getBoundingClientRect();
                return style.display !== 'none' && style.visibility !== 'hidden' && rect.width > 0 && rect.height > 0;
              };
              for (const table of Array.from(document.querySelectorAll('table')).filter(visible).slice(0, maxTables)) {
                const caption = table.querySelector('caption')?.innerText?.trim() || '';
                const rawRows = Array.from(table.querySelectorAll('tr')).map((tr) =>
                  Array.from(tr.querySelectorAll('th,td')).map((cell) => cell.innerText.trim().replace(/\\s+/g, ' '))
                ).filter((row) => row.length > 0);
                if (!rawRows.length) continue;
                const hasHeader = table.querySelector('th') !== null;
                const columns = hasHeader ? rawRows[0].map((c, i) => c || `column_${i + 1}`) : rawRows[0].map((_, i) => `column_${i + 1}`);
                const dataRows = hasHeader ? rawRows.slice(1) : rawRows;
                const rows = dataRows.slice(0, maxRows).map((row) => {
                  const obj = {};
                  columns.forEach((col, i) => { obj[col] = row[i] || ''; });
                  return obj;
                });
                result.push({caption, columns, rows, row_count: rows.length});
              }
              return result;
            }
            """,
            {"maxTables": max_tables, "maxRows": max_rows},
        )
        return {"url": self._page.url, "title": await self._page.title(), "tables": tables, "table_count": len(tables)}

    async def set_date_range(self, start_date: str, end_date: str, submit: bool = True) -> dict:
        """Best-effort date range setter for interactive pages."""
        await self._ensure_browser()
        result = await self._page.evaluate(
            """
            async ({startDate, endDate, submit}) => {
              const sleep = (ms) => new Promise((resolve) => setTimeout(resolve, ms));
              const norm = (value) => (value || '').toString().toLowerCase();
              const labelText = (el) => {
                const id = el.id;
                const label = id ? document.querySelector(`label[for="${CSS.escape(id)}"]`) : null;
                return norm([
                  el.getAttribute('name'),
                  el.getAttribute('id'),
                  el.getAttribute('aria-label'),
                  el.getAttribute('placeholder'),
                  el.getAttribute('title'),
                  label?.innerText
                ].filter(Boolean).join(' '));
              };
              const visible = (el) => {
                const style = window.getComputedStyle(el);
                const rect = el.getBoundingClientRect();
                return style.display !== 'none' && style.visibility !== 'hidden' && rect.width > 0 && rect.height > 0 && !el.disabled;
              };
              const inputs = Array.from(document.querySelectorAll('input')).filter(visible);
              const startHints = ['start', 'from', 'begin', 'date from', 'from date', 'since', 'after'];
              const endHints = ['end', 'to', 'until', 'date to', 'to date', 'before'];
              const dateLike = inputs.filter((el) => {
                const text = labelText(el);
                const type = norm(el.getAttribute('type'));
                const hinted = text.includes('date') || startHints.concat(endHints).some((hint) => text.includes(hint));
                return type === 'date' || ((type === 'text' || type === 'search') && hinted);
              });
              const pick = (hints, exclude) => {
                for (const hint of hints) {
                  const match = dateLike.find((el) => el !== exclude && labelText(el).includes(hint));
                  if (match) return match;
                }
                return dateLike.find((el) => el !== exclude) || null;
              };
              const startInput = pick(startHints, null);
              const endInput = pick(endHints, startInput);
              const setValue = (el, value) => {
                if (!el) return false;
                el.focus();
                el.value = value;
                el.dispatchEvent(new Event('input', {bubbles: true}));
                el.dispatchEvent(new Event('change', {bubbles: true}));
                el.blur();
                return true;
              };
              const filled = [];
              if (setValue(startInput, startDate)) filled.push({field: 'start', selector: labelText(startInput), value: startDate});
              if (setValue(endInput, endDate)) filled.push({field: 'end', selector: labelText(endInput), value: endDate});
              const clickedDates = [];
              if (!filled.length) {
                const clickDate = async (value, field) => {
                  const candidates = Array.from(document.querySelectorAll('button,a,td,[role="button"],[data-date],[aria-label],[title]')).filter(visible);
                  const compact = value.replaceAll('-', '');
                  const day = String(Number(value.slice(-2)));
                  const match = candidates.find((el) => {
                    const text = norm([
                      el.innerText,
                      el.getAttribute('aria-label'),
                      el.getAttribute('title'),
                      el.getAttribute('data-date'),
                      el.getAttribute('datetime')
                    ].filter(Boolean).join(' '));
                    return text.includes(value) || text.includes(compact) || text === day;
                  });
                  if (!match) return false;
                  match.click();
                  clickedDates.push({field, value, selector: norm(match.innerText || match.getAttribute('aria-label') || match.getAttribute('title') || match.getAttribute('data-date'))});
                  await sleep(300);
                  return true;
                };
                await clickDate(startDate, 'start');
                await clickDate(endDate, 'end');
              }
              let clicked = null;
              if (submit) {
                const buttons = Array.from(document.querySelectorAll('button,input[type="submit"],input[type="button"],[role="button"]')).filter(visible);
                const submitHints = ['apply', 'search', 'submit', 'update', 'show', 'filter', 'go'];
                const button = buttons.find((el) => submitHints.some((hint) => norm(el.innerText || el.value || el.getAttribute('aria-label')).includes(hint)));
                if (button) {
                  clicked = norm(button.innerText || button.value || button.getAttribute('aria-label'));
                  button.click();
                  await sleep(800);
                }
              }
              return {ok: filled.length > 0 || clickedDates.length > 0, filled, clicked_dates: clickedDates, clicked, input_count: inputs.length, date_like_count: dateLike.length};
            }
            """,
            {"startDate": start_date, "endDate": end_date, "submit": submit},
        )
        return {"url": self._page.url, "title": await self._page.title(), **result}

    async def extract_tables_for_date_range(
        self,
        start_date: str,
        end_date: str,
        max_tables: int = 8,
        max_rows: int = 100,
    ) -> dict:
        date_result = await self.set_date_range(start_date=start_date, end_date=end_date, submit=True)
        tables = await self.extract_tables(max_tables=max_tables, max_rows=max_rows)
        return {"date_range": date_result, **tables}

    async def scroll(self, direction: str) -> dict:
        await self._ensure_browser()
        if direction == "down":
            await self._page.evaluate("window.scrollBy(0, window.innerHeight)")
        elif direction == "up":
            await self._page.evaluate("window.scrollBy(0, -window.innerHeight)")
        elif direction == "top":
            await self._page.evaluate("window.scrollTo(0, 0)")
        elif direction == "bottom":
            await self._page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
        await asyncio.sleep(0.3)
        return {"ok": True, "url": self._page.url}

    async def screenshot(self, full_page: bool = False, ocr: bool = False) -> dict:
        """Capture a PNG screenshot of the current page, save it, and optionally OCR it.

        Saving to disk (rather than returning base64) keeps tool output small. OCR
        extracts text locked inside the rendered image (charts, canvas, etc.).
        """
        await self._ensure_browser()
        png = await self._page.screenshot(full_page=full_page)

        import hashlib
        import os

        out_dir = Path(os.getenv("FOOTNOTE_SOURCE_CACHE", "~/.footnote-mcp/source_cache")).expanduser() / "screenshots"
        out_dir.mkdir(parents=True, exist_ok=True)
        name = hashlib.sha256(f"{self._page.url}|{full_page}".encode("utf-8")).hexdigest()[:16] + ".png"
        path = out_dir / name
        path.write_bytes(png)

        result = {
            "url": self._page.url,
            "title": await self._page.title(),
            "path": str(path),
            "bytes": len(png),
            "full_page": full_page,
        }
        if ocr:
            result["ocr_text"] = _ocr_png_bytes(png)
        return result

    async def close(self):
        if self._browser:
            await self._browser.close()
        if self._playwright:
            await self._playwright.stop()


async def _test():
    """Smoke test: browser navigation, snapshot, click, extract, scroll."""
    print("=== tools_browser tests ===\n")
    b = WebBrowser(headed=False)

    # 1. navigate
    print("1. navigate...")
    r = await b.navigate("https://www.iana.org/domains/reserved")
    assert r["url"].startswith("http"), f"bad url after navigate: {r['url']}"
    assert r["title"], "empty title"
    print(f"   ✓ {r['title']}\n")

    # 2. snapshot
    print("2. snapshot...")
    snap = await b.snapshot()
    assert snap["url"].startswith("http"), f"bad url in snapshot: {snap['url']}"
    assert snap["element_count"] >= 0, f"bad element count: {snap['element_count']}"
    assert snap["visible_text"], "visible_text is empty"
    assert len(snap["elements"]) == snap["element_count"], "element count mismatch"
    link_refs = [e["ref"] for e in snap["elements"] if e["role"] == "link"]
    print(f"   ✓ {snap['element_count']} elements, {len(link_refs)} links\n")

    # 3. click a link
    if link_refs:
        print(f"3. click({link_refs[0]})...")
        click_result = await b.click(link_refs[0])
        assert click_result["ok"], f"click failed: {click_result.get('error')}"
        print(f"   ✓ navigated to {click_result['url'][:60]}\n")
    else:
        print("3. skipped (no links on page)\n")

    # 4. extract visible text
    print("4. extract('visible')...")
    ext = await b.extract("visible")
    assert ext["text"], "extracted text is empty"
    print(f"   ✓ {len(ext['text'])} chars\n")

    # 5. scroll
    print("5. scroll(down)...")
    sc = await b.scroll("down")
    assert sc["ok"], f"scroll failed: {sc}"
    print("   ✓\n")

    # 6. bad ref returns error
    print("6. click(bad ref)...")
    bad_click = await b.click("@e99999")
    assert not bad_click["ok"], f"expected error for bad ref, got {bad_click}"
    assert "error" in bad_click
    print(f"   ✓ error: {bad_click['error'][:60]}\n")

    # 7. type in a textbox (navigate to search page)
    print("7. type in searchbox...")
    await b.navigate("https://www.iana.org/domains/reserved")
    snap2 = await b.snapshot()
    searchboxes = [e for e in snap2["elements"] if e["role"] == "searchbox"]
    if searchboxes:
        type_result = await b.type(searchboxes[0]["ref"], "hello", submit=False)
        assert type_result["ok"], f"type failed: {type_result.get('error')}"
        print(f"   ✓ typed 'hello' into {searchboxes[0]['ref']}\n")
    else:
        print("   ⚠ no searchbox found\n")

    await b.close()
    print("=== all browser tests passed ===")


if __name__ == "__main__":
    asyncio.run(_test())
