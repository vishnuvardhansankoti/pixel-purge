"""EXPERIMENTAL targeted deletion via Playwright browser automation.

> [!WARNING]
> Google frequently changes the Photos web DOM; selectors break without notice.
> Filename search is unreliable [M5], so this driver navigates to each item by its
> Google Photos URL (captured at upload/classification time as `cloud_media_id`)
> rather than searching by filename. Items without a known cloud URL are skipped
> and reported — never guessed at. Treat this as best-effort and verify results.

Requires the optional `[browser]` extra (Playwright) and a logged-in session.
Not covered by automated tests (needs a real Google account); the safe primary
path is the exported deletion manifest.
"""

from __future__ import annotations

from dataclasses import dataclass

from rich.console import Console

from ..database import Database

console = Console()


@dataclass
class BrowserDeleteResult:
    deleted: int = 0
    skipped_no_url: int = 0
    errors: int = 0


async def delete_flagged(db: Database, headless: bool = False,
                         delay_ms: int = 4000) -> BrowserDeleteResult:
    """Open Google Photos and move each DELETE-flagged item (with a known cloud
    URL) to trash. Returns a result summary."""
    from playwright.async_api import async_playwright

    deletions = db.get_deletions()
    result = BrowserDeleteResult()
    targets = [d for d in deletions if d.cloud_media_id]
    result.skipped_no_url = len(deletions) - len(targets)

    if not targets:
        console.print("[yellow]Browser delete:[/yellow] no items have a known "
                      "Google Photos URL to target; nothing to do.")
        return result

    console.print(f"[bold red]EXPERIMENTAL[/bold red] browser deletion of "
                  f"{len(targets)} item(s). Verify results afterward.")

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=headless)
        context = await browser.new_context()
        page = await context.new_page()
        for item in targets:
            try:
                await page.goto(f"https://photos.google.com/lr/photo/{item.cloud_media_id}")
                await page.wait_for_timeout(delay_ms)
                # Trash via keyboard shortcut (# in Google Photos), then confirm.
                await page.keyboard.press("#")
                await page.wait_for_timeout(1000)
                confirm = await page.query_selector('button:has-text("Move to trash")')
                if confirm:
                    await confirm.click()
                    db.set_keeper_status(item.id, "DELETE")  # already DELETE; touch mtime
                    result.deleted += 1
                await page.wait_for_timeout(delay_ms)
            except Exception as e:  # noqa: BLE001
                result.errors += 1
                console.print(f"[yellow]browser delete error[/yellow] "
                              f"{item.filename}: {e}")
        await browser.close()

    db.commit()
    return result
