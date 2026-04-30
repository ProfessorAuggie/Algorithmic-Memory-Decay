"""WSGI entrypoint for the Algorithmic Memory Decay simulation gallery."""

from __future__ import annotations

import html
import mimetypes
from pathlib import Path
from typing import Iterable


BASE_DIR = Path(__file__).resolve().parent
SCREENSHOTS_DIR = BASE_DIR / "screenshots"

SIMULATION_SECTIONS = [
    {
        "title": "Simulation 1: Recommendation Accuracy under Memory Decay",
        "description": "Accuracy trends when older interactions decay over time.",
        "files": ["simulation_1_accuracy_3d.png", "simulation_1_accuracy_2d.png"],
    },
    {
        "title": "Simulation 2: Privacy vs Engagement Trade-off",
        "description": "How the model balances engagement with exposure risk.",
        "files": ["simulation_2_privacy_engagement.png", "simulation_2_contours.png"],
    },
    {
        "title": "Simulation 3: Comparative Performance Analysis",
        "description": "A comparison against baseline strategies and a summary table.",
        "files": ["simulation_3_comparative.png", "simulation_3_table.png"],
    },
    {
        "title": "Simulation 4: Entropy and System Dynamics",
        "description": "How entropy, decay, and recommendation dynamics evolve together.",
        "files": ["simulation_4_dynamics.png"],
    },
]


def _section_anchor(title: str) -> str:
    return title.lower().split(":", 1)[0].replace(" ", "-").replace("/", "-")


def _build_html() -> bytes:
    nav_links = []
    cards = []

    for section in SIMULATION_SECTIONS:
        anchor = _section_anchor(section["title"])
        nav_links.append(
            f'<a class="nav-link" href="#{anchor}">{html.escape(section["title"].split(":", 1)[0])}</a>'
        )

        figures = []
        for filename in section["files"]:
            candidate = SCREENSHOTS_DIR / filename
            if not candidate.exists():
                continue
            label = html.escape(filename.replace("_", " ").replace(".png", "").title())
            figures.append(
                f"""
                <figure class="figure-card">
                  <img src="/screenshots/{html.escape(filename)}" alt="{label}">
                  <figcaption>{label}</figcaption>
                </figure>
                """
            )

        cards.append(
            f"""
            <section class="section-card" id="{anchor}">
              <div class="section-copy">
                <h2>{html.escape(section['title'])}</h2>
                <p>{html.escape(section['description'])}</p>
              </div>
              <div class="figure-grid">
                {''.join(figures)}
              </div>
            </section>
            """
        )

    body = f"""
    <!doctype html>
    <html lang="en">
      <head>
        <meta charset="utf-8">
        <meta name="viewport" content="width=device-width, initial-scale=1">
        <title>Algorithmic Memory Decay | Simulation Gallery</title>
        <style>
          :root {{
            color-scheme: dark;
            --bg: #0b1020;
            --panel: rgba(16, 24, 40, 0.86);
            --panel-border: rgba(148, 163, 184, 0.18);
            --text: #e5eefc;
            --muted: #9fb2d4;
            --accent: #7dd3fc;
          }}

          * {{ box-sizing: border-box; }}
          body {{
            margin: 0;
            font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
            color: var(--text);
            background:
              radial-gradient(circle at top left, rgba(125, 211, 252, 0.18), transparent 30%),
              radial-gradient(circle at top right, rgba(251, 191, 36, 0.12), transparent 28%),
              linear-gradient(180deg, #060913 0%, #0b1020 45%, #0a0f1d 100%);
            min-height: 100vh;
          }}

          .wrap {{ max-width: 1280px; margin: 0 auto; padding: 40px 20px 64px; }}
          .hero {{
            background: linear-gradient(135deg, rgba(30, 41, 59, 0.92), rgba(15, 23, 42, 0.84));
            border: 1px solid var(--panel-border);
            border-radius: 24px;
            padding: 32px;
            box-shadow: 0 24px 80px rgba(0, 0, 0, 0.35);
            margin-bottom: 24px;
          }}
          .eyebrow {{
            display: inline-block;
            padding: 6px 12px;
            border-radius: 999px;
            background: rgba(125, 211, 252, 0.12);
            color: var(--accent);
            font-size: 12px;
            letter-spacing: 0.12em;
            text-transform: uppercase;
          }}
          h1 {{ margin: 16px 0 12px; font-size: clamp(2rem, 4vw, 4rem); line-height: 1.02; }}
          .lead {{ margin: 0; max-width: 820px; color: var(--muted); font-size: 1.05rem; line-height: 1.7; }}
          .cta-row {{ display: flex; flex-wrap: wrap; gap: 12px; margin-top: 22px; }}
          .cta {{
            display: inline-flex;
            align-items: center;
            justify-content: center;
            padding: 11px 16px;
            border-radius: 999px;
            text-decoration: none;
            font-weight: 600;
            border: 1px solid transparent;
          }}
          .cta.primary {{ background: linear-gradient(135deg, var(--accent), #60a5fa); color: #06111f; }}
          .cta.secondary {{ background: rgba(148, 163, 184, 0.12); color: var(--text); border-color: rgba(148, 163, 184, 0.18); }}
          .nav-row {{ display: flex; flex-wrap: wrap; gap: 10px; margin-top: 18px; }}
          .nav-link {{
            display: inline-flex;
            align-items: center;
            padding: 8px 12px;
            border-radius: 999px;
            background: rgba(15, 23, 42, 0.72);
            border: 1px solid rgba(125, 211, 252, 0.2);
            color: #e5eefc;
            text-decoration: none;
            font-size: 0.92rem;
          }}
          .meta {{ display: flex; flex-wrap: wrap; gap: 12px; margin-top: 20px; }}
          .meta span {{
            padding: 8px 12px;
            border-radius: 999px;
            background: rgba(148, 163, 184, 0.12);
            border: 1px solid var(--panel-border);
            color: #d7e2f5;
            font-size: 0.92rem;
          }}
          .section-card {{
            background: var(--panel);
            border: 1px solid var(--panel-border);
            border-radius: 22px;
            padding: 24px;
            margin-top: 18px;
          }}
          .section-copy h2 {{ margin: 0 0 8px; font-size: 1.4rem; }}
          .section-copy p {{ margin: 0 0 18px; color: var(--muted); line-height: 1.65; }}
          .figure-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
            gap: 16px;
          }}
          .figure-card {{
            margin: 0;
            border-radius: 18px;
            overflow: hidden;
            background: rgba(15, 23, 42, 0.92);
            border: 1px solid rgba(148, 163, 184, 0.16);
          }}
          .figure-card img {{ display: block; width: 100%; height: auto; background: #0f172a; }}
          .figure-card figcaption {{
            padding: 12px 14px 14px;
            color: #dbe6fb;
            font-size: 0.95rem;
            border-top: 1px solid rgba(148, 163, 184, 0.12);
          }}
          .footer {{ margin-top: 24px; color: var(--muted); font-size: 0.92rem; }}
          a {{ color: var(--accent); }}
        </style>
      </head>
      <body>
        <main class="wrap">
          <section class="hero">
            <span class="eyebrow">Algorithmic Memory Decay</span>
            <h1>Simulation Gallery</h1>
            <p class="lead">
              This page surfaces the generated TEMD / EG-AMD simulation outputs directly in the browser.
              Use it to inspect the accuracy, privacy, comparative, and dynamics figures without leaving the website.
            </p>
            <div class="cta-row">
              <a class="cta primary" href="/summary.json">Download summary JSON</a>
              <a class="cta secondary" href="#simulation-1">Jump to simulations</a>
            </div>
            <nav class="nav-row" aria-label="Simulation sections">
              {''.join(nav_links)}
            </nav>
            <div class="meta">
              <span>4 simulations</span>
              <span>7 generated figures</span>
              <span>Served from screenshots</span>
            </div>
          </section>
          {''.join(cards)}
          <p class="footer">
            If a figure is missing, rerun <a href="/">the simulations</a> locally and refresh this page.
          </p>
        </main>
      </body>
    </html>
    """

    return body.encode("utf-8")


def _serve_file(path: Path) -> tuple[str, list[tuple[str, str]], bytes]:
    content_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
    body = path.read_bytes()
    headers = [
        ("Content-Type", content_type),
        ("Content-Length", str(len(body))),
        ("Cache-Control", "public, max-age=3600"),
    ]
    return "200 OK", headers, body


def app(environ, start_response) -> Iterable[bytes]:
    path = environ.get("PATH_INFO", "/")

    if path == "/summary.json":
        summary_path = SCREENSHOTS_DIR / "simulation_summary.json"
        if summary_path.exists() and summary_path.is_file():
            body = summary_path.read_bytes()
            headers = [
                ("Content-Type", "application/json; charset=utf-8"),
                ("Content-Length", str(len(body))),
                ("Content-Disposition", 'attachment; filename="simulation_summary.json"'),
            ]
            start_response("200 OK", headers)
            return [body]

    if path == "/":
        body = _build_html()
        headers = [
            ("Content-Type", "text/html; charset=utf-8"),
            ("Content-Length", str(len(body))),
        ]
        start_response("200 OK", headers)
        return [body]

    if path.startswith("/screenshots/"):
        filename = Path(path.removeprefix("/screenshots/")).name
        candidate = SCREENSHOTS_DIR / filename
        if candidate.exists() and candidate.is_file():
            status, headers, body = _serve_file(candidate)
            start_response(status, headers)
            return [body]

    body = b"Not Found"
    headers = [
        ("Content-Type", "text/plain; charset=utf-8"),
        ("Content-Length", str(len(body))),
    ]
    start_response("404 Not Found", headers)
    return [body]
