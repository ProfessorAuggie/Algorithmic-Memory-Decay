"""WSGI entrypoint for the Algorithmic Memory Decay simulation gallery."""

from __future__ import annotations

import base64
import html
import json
import mimetypes
from pathlib import Path
from typing import Any, Iterable

import matplotlib

matplotlib.use("Agg")
import numpy as np


BASE_DIR = Path(__file__).resolve().parent
SCREENSHOTS_DIR = BASE_DIR / "screenshots"


def _section_anchor(title: str) -> str:
    return title.lower().split(":", 1)[0].replace(" ", "-").replace("/", "-")


def _safe_int(value: Any, default: int, minimum: int | None = None, maximum: int | None = None) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        parsed = default

    if minimum is not None:
        parsed = max(minimum, parsed)
    if maximum is not None:
        parsed = min(maximum, parsed)
    return parsed


def _safe_float(value: Any, default: float, minimum: float | None = None, maximum: float | None = None) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        parsed = default

    if minimum is not None:
        parsed = max(minimum, parsed)
    if maximum is not None:
        parsed = min(maximum, parsed)
    return parsed


def _read_request_body(environ: dict[str, Any]) -> bytes:
    length = _safe_int(environ.get("CONTENT_LENGTH"), 0, minimum=0)
    if length <= 0:
        return b""
    return environ["wsgi.input"].read(length)


def _json_response(start_response, status: str, payload: dict[str, Any]) -> Iterable[bytes]:
    body = json.dumps(payload).encode("utf-8")
    headers = [
        ("Content-Type", "application/json; charset=utf-8"),
        ("Content-Length", str(len(body))),
        ("Cache-Control", "no-store"),
    ]
    start_response(status, headers)
    return [body]


def _serve_file(path: Path) -> tuple[str, list[tuple[str, str]], bytes]:
    content_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
    body = path.read_bytes()
    headers = [
        ("Content-Type", content_type),
        ("Content-Length", str(len(body))),
        ("Cache-Control", "public, max-age=3600"),
    ]
    return "200 OK", headers, body


def _data_uri(path: str) -> str:
    image_path = Path(path)
    content_type = mimetypes.guess_type(image_path.name)[0] or "image/png"
    encoded = base64.b64encode(image_path.read_bytes()).decode("ascii")
    return f"data:{content_type};base64,{encoded}"


def _live_gallery_cards() -> str:
    sections = [
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

    cards = []
    for section in sections:
        figures = []
        for filename in section["files"]:
            candidate = SCREENSHOTS_DIR / filename
            if not candidate.exists():
                continue
            label = html.escape(filename.replace("_", " ").replace(".png", "").title())
            figures.append(
                f"""
                <figure class="figure-card">
                  <img data-filename="{html.escape(filename)}" src="/screenshots/{html.escape(filename)}" alt="{label}">
                  <figcaption>{label}</figcaption>
                </figure>
                """
            )

        cards.append(
            f"""
            <section class="section-card" id="{_section_anchor(section['title'])}">
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

    return "".join(cards)


def _build_html() -> bytes:
    nav_links = []
    for title in [
        "Simulation 1: Recommendation Accuracy under Memory Decay",
        "Simulation 2: Privacy vs Engagement Trade-off",
        "Simulation 3: Comparative Performance Analysis",
        "Simulation 4: Entropy and System Dynamics",
    ]:
        nav_links.append(
            f'<a class="nav-link" href="#{_section_anchor(title)}">{html.escape(title.split(":", 1)[0])}</a>'
        )

    body = f"""
    <!doctype html>
    <html lang="en">
      <head>
        <meta charset="utf-8">
        <meta name="viewport" content="width=device-width, initial-scale=1">
        <title>Algorithmic Memory Decay | Live Simulation Lab</title>
        <style>
          :root {{
            color-scheme: dark;
            --panel: rgba(16, 24, 40, 0.86);
            --panel-border: rgba(148, 163, 184, 0.18);
            --text: #e5eefc;
            --muted: #9fb2d4;
            --accent: #7dd3fc;
            --accent-2: #fbbf24;
            --ok: #34d399;
            --danger: #fb7185;
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

          .wrap {{ max-width: 1360px; margin: 0 auto; padding: 32px 18px 64px; }}
          .hero {{
            background: linear-gradient(135deg, rgba(30, 41, 59, 0.92), rgba(15, 23, 42, 0.84));
            border: 1px solid var(--panel-border);
            border-radius: 24px;
            padding: 28px;
            box-shadow: 0 24px 80px rgba(0, 0, 0, 0.35);
            margin-bottom: 18px;
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
          h1 {{ margin: 16px 0 10px; font-size: clamp(2rem, 4vw, 4rem); line-height: 1.02; }}
          .lead {{ margin: 0; max-width: 980px; color: var(--muted); font-size: 1.02rem; line-height: 1.7; }}
          .meta {{ display: flex; flex-wrap: wrap; gap: 10px; margin-top: 18px; }}
          .meta span, .pill {{
            padding: 8px 12px;
            border-radius: 999px;
            background: rgba(148, 163, 184, 0.12);
            border: 1px solid var(--panel-border);
            color: #d7e2f5;
            font-size: 0.92rem;
          }}
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
          .layout {{
            display: grid;
            grid-template-columns: 380px minmax(0, 1fr);
            gap: 18px;
            align-items: start;
            margin-top: 18px;
          }}
          .panel {{
            background: var(--panel);
            border: 1px solid var(--panel-border);
            border-radius: 22px;
            padding: 20px;
          }}
          .panel h2 {{ margin: 0 0 10px; font-size: 1.15rem; }}
          .controls {{ display: grid; gap: 14px; }}
          .field-group {{ display: grid; gap: 10px; }}
          .field-group label {{ font-size: 0.92rem; color: #dbe6fb; }}
          .field-group input, .field-group select {{
            width: 100%;
            border-radius: 12px;
            border: 1px solid rgba(148, 163, 184, 0.24);
            background: rgba(15, 23, 42, 0.88);
            color: var(--text);
            padding: 11px 12px;
            font-size: 0.98rem;
            outline: none;
          }}
          .field-group input:focus, .field-group select:focus {{ border-color: rgba(125, 211, 252, 0.7); }}
          .sim-panel {{ display: none; gap: 12px; }}
          .sim-panel.active {{ display: grid; }}
          .grid-2 {{ display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 12px; }}
          .button-row {{ display: flex; flex-wrap: wrap; gap: 10px; margin-top: 4px; }}
          .button {{
            appearance: none;
            border: 0;
            border-radius: 999px;
            padding: 11px 16px;
            font-weight: 700;
            cursor: pointer;
          }}
          .button.primary {{ background: linear-gradient(135deg, var(--accent), #60a5fa); color: #06111f; }}
          .button.secondary {{ background: rgba(148, 163, 184, 0.12); color: var(--text); border: 1px solid rgba(148, 163, 184, 0.18); }}
          .status {{ margin-top: 12px; font-size: 0.92rem; color: var(--muted); }}
          .status strong {{ color: var(--text); }}
          .results {{ display: grid; gap: 18px; }}
          .summary-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 12px; }}
          .metric-card {{
            background: rgba(15, 23, 42, 0.86);
            border: 1px solid rgba(148, 163, 184, 0.16);
            border-radius: 18px;
            padding: 14px;
          }}
          .metric-card .label {{ display: block; color: var(--muted); font-size: 0.82rem; margin-bottom: 8px; }}
          .metric-card .value {{ font-size: 1.05rem; font-weight: 700; line-height: 1.35; white-space: pre-wrap; }}
          .figure-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); gap: 16px; }}
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
          pre {{
            margin: 0;
            overflow: auto;
            background: rgba(15, 23, 42, 0.88);
            border: 1px solid rgba(148, 163, 184, 0.16);
            border-radius: 18px;
            padding: 16px;
            color: #dce7fb;
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
          .footer {{ margin-top: 24px; color: var(--muted); font-size: 0.92rem; }}
          .hidden {{ display: none !important; }}
          a {{ color: var(--accent); }}
          @media (max-width: 980px) {{
            .layout {{ grid-template-columns: 1fr; }}
          }}
        </style>
      </head>
      <body>
        <main class="wrap">
          <section class="hero">
            <span class="eyebrow">Algorithmic Memory Decay</span>
            <h1>Live Simulation Lab</h1>
            <p class="lead">
              Change the parameters, run a simulation in the browser, and update the figures immediately.
              The latest outputs also refresh the gallery below, so you can compare the live results with the saved screenshots.
            </p>
            <div class="nav-row">
              <a class="nav-link" href="#live-run">Live run</a>
              <a class="nav-link" href="#latest-gallery">Latest gallery</a>
              <a class="nav-link" href="/summary.json">Download summary JSON</a>
            </div>
            <div class="meta">
              <span>4 simulations</span>
              <span>Editable parameters</span>
              <span>Real-time results</span>
            </div>
          </section>

          <section class="layout" id="live-run">
            <section class="panel">
              <h2>Run a simulation</h2>
              <div class="controls">
                <div class="field-group">
                  <label for="simulation">Simulation</label>
                  <select id="simulation" name="simulation">
                    <option value="1">Simulation 1 - Recommendation Accuracy</option>
                    <option value="2">Simulation 2 - Privacy vs Engagement</option>
                    <option value="3">Simulation 3 - Comparative Analysis</option>
                    <option value="4">Simulation 4 - Entropy and Dynamics</option>
                  </select>
                </div>

                <div class="sim-panel active" data-sim-panel="1">
                  <div class="grid-2">
                    <div class="field-group"><label for="sim1_users">Users</label><input id="sim1_users" value="12" type="number" min="1" step="1"></div>
                    <div class="field-group"><label for="sim1_items">Items</label><input id="sim1_items" value="12" type="number" min="2" step="1"></div>
                    <div class="field-group"><label for="sim1_feature_dim">Feature dim</label><input id="sim1_feature_dim" value="6" type="number" min="2" step="1"></div>
                    <div class="field-group"><label for="sim1_time_steps">Time steps</label><input id="sim1_time_steps" value="12" type="number" min="2" step="1"></div>
                    <div class="field-group"><label for="sim1_decay_start">Decay start</label><input id="sim1_decay_start" value="0.03" type="number" min="0" step="0.01"></div>
                    <div class="field-group"><label for="sim1_decay_end">Decay end</label><input id="sim1_decay_end" value="0.25" type="number" min="0" step="0.01"></div>
                    <div class="field-group"><label for="sim1_decay_points">Decay points</label><input id="sim1_decay_points" value="8" type="number" min="2" step="1"></div>
                    <div class="field-group"><label for="sim1_seed">Random seed</label><input id="sim1_seed" value="42" type="number" step="1"></div>
                  </div>
                </div>

                <div class="sim-panel" data-sim-panel="2">
                  <div class="grid-2">
                    <div class="field-group"><label for="sim2_users">Users</label><input id="sim2_users" value="18" type="number" min="1" step="1"></div>
                    <div class="field-group"><label for="sim2_items">Items</label><input id="sim2_items" value="16" type="number" min="2" step="1"></div>
                    <div class="field-group"><label for="sim2_feature_dim">Feature dim</label><input id="sim2_feature_dim" value="6" type="number" min="2" step="1"></div>
                    <div class="field-group"><label for="sim2_seed">Random seed</label><input id="sim2_seed" value="42" type="number" step="1"></div>
                    <div class="field-group"><label for="sim2_decay_start">Decay start</label><input id="sim2_decay_start" value="0.03" type="number" min="0" step="0.01"></div>
                    <div class="field-group"><label for="sim2_decay_end">Decay end</label><input id="sim2_decay_end" value="0.4" type="number" min="0" step="0.01"></div>
                    <div class="field-group"><label for="sim2_decay_points">Decay points</label><input id="sim2_decay_points" value="8" type="number" min="2" step="1"></div>
                    <div class="field-group"><label for="sim2_engagement_start">Engagement start</label><input id="sim2_engagement_start" value="0.2" type="number" min="0" step="0.05"></div>
                    <div class="field-group"><label for="sim2_engagement_end">Engagement end</label><input id="sim2_engagement_end" value="0.8" type="number" min="0" step="0.05"></div>
                    <div class="field-group"><label for="sim2_engagement_points">Engagement points</label><input id="sim2_engagement_points" value="8" type="number" min="2" step="1"></div>
                  </div>
                </div>

                <div class="sim-panel" data-sim-panel="3">
                  <div class="grid-2">
                    <div class="field-group"><label for="sim3_users">Users</label><input id="sim3_users" value="16" type="number" min="1" step="1"></div>
                    <div class="field-group"><label for="sim3_items">Items</label><input id="sim3_items" value="18" type="number" min="2" step="1"></div>
                    <div class="field-group"><label for="sim3_feature_dim">Feature dim</label><input id="sim3_feature_dim" value="8" type="number" min="2" step="1"></div>
                    <div class="field-group"><label for="sim3_time_steps">Time steps</label><input id="sim3_time_steps" value="12" type="number" min="2" step="1"></div>
                    <div class="field-group"><label for="sim3_seed">Random seed</label><input id="sim3_seed" value="42" type="number" step="1"></div>
                  </div>
                </div>

                <div class="sim-panel" data-sim-panel="4">
                  <div class="grid-2">
                    <div class="field-group"><label for="sim4_users">Users</label><input id="sim4_users" value="18" type="number" min="1" step="1"></div>
                    <div class="field-group"><label for="sim4_time_steps">Time steps</label><input id="sim4_time_steps" value="24" type="number" min="2" step="1"></div>
                    <div class="field-group"><label for="sim4_seed">Random seed</label><input id="sim4_seed" value="42" type="number" step="1"></div>
                  </div>
                </div>

                <div class="button-row">
                  <button class="button primary" type="button" id="run-button">Run in real time</button>
                  <a class="button secondary" href="/summary.json" style="text-decoration:none; display:inline-flex; align-items:center;">Download summary</a>
                </div>
                <div class="status" id="run-status">Pick a simulation and press <strong>Run in real time</strong>.</div>
              </div>
            </section>

            <section class="panel results" aria-live="polite">
              <h2>Live output</h2>
              <div class="summary-grid" id="summary-grid">
                <div class="metric-card"><span class="label">Status</span><span class="value">Ready to run</span></div>
              </div>
              <div class="figure-grid" id="live-figures"></div>
              <pre id="raw-json">No live results yet.</pre>
            </section>
          </section>

          <section class="panel" id="latest-gallery" style="margin-top:18px;">
            <h2>Latest gallery</h2>
            <p class="lead" style="max-width:none;">These are the saved screenshot outputs. They update after each run so you can keep a stable gallery while experimenting with parameters.</p>
            <div class="figure-grid">
              {_live_gallery_cards()}
            </div>
          </section>

          <section class="section-card">
            <div class="section-copy">
              <h2>Quick tips</h2>
              <p>Simulation 1 and 2 are the fastest way to see parameter changes reflected immediately. Simulation 3 is heavier, and Simulation 4 is best for seeing entropy and decay dynamics shift over time.</p>
            </div>
          </section>

          <p class="footer">If a figure looks stale, run a new simulation and the page will refresh both the live output and the gallery images.</p>
        </main>

        <script>
          const simulationSelect = document.getElementById('simulation');
          const panels = Array.from(document.querySelectorAll('.sim-panel'));
          const runButton = document.getElementById('run-button');
          const runStatus = document.getElementById('run-status');
          const summaryGrid = document.getElementById('summary-grid');
          const liveFigures = document.getElementById('live-figures');
          const rawJson = document.getElementById('raw-json');

          function escapeHtml(value) {{
            return String(value)
              .replaceAll('&', '&amp;')
              .replaceAll('<', '&lt;')
              .replaceAll('>', '&gt;')
              .replaceAll('"', '&quot;')
              .replaceAll("'", '&#39;');
          }}

          function showActivePanel() {{
            const active = simulationSelect.value;
            panels.forEach((panel) => panel.classList.toggle('active', panel.dataset.simPanel === active));
          }}

          function valueById(id) {{
            const element = document.getElementById(id);
            return element ? element.value : '';
          }}

          function readPayload() {{
            const simulation = simulationSelect.value;
            const payload = {{ simulation }};

            if (simulation === '1') {{
              payload.n_users = Number(valueById('sim1_users'));
              payload.n_items = Number(valueById('sim1_items'));
              payload.feature_dim = Number(valueById('sim1_feature_dim'));
              payload.time_steps = Number(valueById('sim1_time_steps'));
              payload.decay_start = Number(valueById('sim1_decay_start'));
              payload.decay_end = Number(valueById('sim1_decay_end'));
              payload.decay_points = Number(valueById('sim1_decay_points'));
              payload.random_seed = Number(valueById('sim1_seed'));
            }} else if (simulation === '2') {{
              payload.n_users = Number(valueById('sim2_users'));
              payload.n_items = Number(valueById('sim2_items'));
              payload.feature_dim = Number(valueById('sim2_feature_dim'));
              payload.decay_start = Number(valueById('sim2_decay_start'));
              payload.decay_end = Number(valueById('sim2_decay_end'));
              payload.decay_points = Number(valueById('sim2_decay_points'));
              payload.engagement_start = Number(valueById('sim2_engagement_start'));
              payload.engagement_end = Number(valueById('sim2_engagement_end'));
              payload.engagement_points = Number(valueById('sim2_engagement_points'));
              payload.random_seed = Number(valueById('sim2_seed'));
            }} else if (simulation === '3') {{
              payload.n_users = Number(valueById('sim3_users'));
              payload.n_items = Number(valueById('sim3_items'));
              payload.feature_dim = Number(valueById('sim3_feature_dim'));
              payload.time_steps = Number(valueById('sim3_time_steps'));
              payload.random_seed = Number(valueById('sim3_seed'));
            }} else {{
              payload.n_users = Number(valueById('sim4_users'));
              payload.time_steps = Number(valueById('sim4_time_steps'));
              payload.random_seed = Number(valueById('sim4_seed'));
            }}

            return payload;
          }}

          function metricCard(label, value) {{
            return `<div class="metric-card"><span class="label">${{escapeHtml(label)}}</span><span class="value">${{escapeHtml(value)}}</span></div>`;
          }}

          function renderSummary(data) {{
            const summary = data.summary || {{}};
            const cards = [];

            cards.push(metricCard('Simulation', data.title || data.simulation || 'Unknown'));

            Object.entries(summary).forEach(([key, value]) => {{
              if (typeof value === 'object' && value !== null && !Array.isArray(value)) {{
                return;
              }}
              cards.push(metricCard(key.replaceAll('_', ' '), Array.isArray(value) ? value.join(', ') : String(value)));
            }});

            if (data.simulation === 'simulation_3' && summary.final_scores) {{
              cards.push(metricCard('Best accuracy', bestScore(summary.final_scores, 'Accuracy')));
              cards.push(metricCard('Best privacy', bestScore(summary.final_scores, 'Privacy')));
              cards.push(metricCard('Best autonomy', bestScore(summary.final_scores, 'Autonomy')));
            }}

            summaryGrid.innerHTML = cards.join('');
          }}

          function bestScore(finalScores, metric) {{
            let bestName = '';
            let bestValue = -Infinity;
            Object.entries(finalScores).forEach(([name, metrics]) => {{
              const value = Number(metrics[metric]);
              if (value > bestValue) {{
                bestValue = value;
                bestName = `${{name}} (${value.toFixed(3)})`;
              }}
            }});
            return bestName;
          }}

          function renderFigures(data) {{
            const figures = data.figures || [];
            liveFigures.innerHTML = figures.map((figure) => `
              <figure class="figure-card">
                <img src="${{figure.data_uri}}" alt="${{escapeHtml(figure.label)}}">
                <figcaption>${{escapeHtml(figure.label)}}</figcaption>
              </figure>
            `).join('');
          }}

          function refreshGallery(data) {{
            const runId = data.run_id || Date.now().toString();
            (data.figures || []).forEach((figure) => {{
              const staticImage = document.querySelector(`img[data-filename="${{CSS.escape(figure.filename)}}"]`);
              if (staticImage) {{
                staticImage.src = `${{figure.static_url}}?v=${{runId}}`;
              }}
            }});
          }}

          async function runSimulation() {{
            runButton.disabled = true;
            runStatus.innerHTML = 'Running simulation...';
            summaryGrid.innerHTML = metricCard('Status', 'Running...');
            liveFigures.innerHTML = '';
            rawJson.textContent = 'Waiting for response...';

            try {{
              const response = await fetch('/run', {{
                method: 'POST',
                headers: {{ 'Content-Type': 'application/json' }},
                body: JSON.stringify(readPayload()),
              }});
              const data = await response.json();

              if (!response.ok) {{
                throw new Error(data.error || 'Simulation run failed');
              }}

              runStatus.innerHTML = `Completed ${{escapeHtml(data.title || data.simulation)}}.`;
              renderSummary(data);
              renderFigures(data);
              refreshGallery(data);
              rawJson.textContent = JSON.stringify(data.summary, null, 2);
            }} catch (error) {{
              runStatus.innerHTML = `<span style="color: var(--danger);">${{escapeHtml(error.message)}}</span>`;
              summaryGrid.innerHTML = metricCard('Error', error.message);
              rawJson.textContent = error.stack || error.message;
            }} finally {{
              runButton.disabled = false;
            }}
          }}

          simulationSelect.addEventListener('change', showActivePanel);
          runButton.addEventListener('click', runSimulation);
          showActivePanel();
        </script>
      </body>
    </html>
    """

    return body.encode("utf-8")


def _run_simulation(payload: dict[str, Any]) -> dict[str, Any]:
    from simulations.test_cases import (
        simulation_1_accuracy_under_decay,
        simulation_2_privacy_engagement_tradeoff,
        simulation_3_comparative_analysis,
        simulation_4_entropy_and_dynamics,
    )

    simulation = str(payload.get("simulation", "1"))

    if simulation == "1":
        decay_rates = np.linspace(
            _safe_float(payload.get("decay_start"), 0.03, minimum=0.0),
            _safe_float(payload.get("decay_end"), 0.25, minimum=0.0),
            _safe_int(payload.get("decay_points"), 8, minimum=2, maximum=50),
        )
        result = simulation_1_accuracy_under_decay(
            n_users=_safe_int(payload.get("n_users"), 12, minimum=1, maximum=40),
            n_items=_safe_int(payload.get("n_items"), 12, minimum=2, maximum=40),
            feature_dim=_safe_int(payload.get("feature_dim"), 6, minimum=2, maximum=20),
            time_steps=_safe_int(payload.get("time_steps"), 12, minimum=2, maximum=30),
            decay_rates=decay_rates,
            random_seed=_safe_int(payload.get("random_seed"), 42, minimum=0, maximum=9999),
        )
        figures = [
            {
                "filename": Path(result["figure_3d"]).name,
                "label": "Simulation 1 - 3D Accuracy Surface",
                "static_url": f"/screenshots/{Path(result['figure_3d']).name}",
                "data_uri": _data_uri(result["figure_3d"]),
            },
            {
                "filename": Path(result["figure_2d"]).name,
                "label": "Simulation 1 - 2D Accuracy Trajectories",
                "static_url": f"/screenshots/{Path(result['figure_2d']).name}",
                "data_uri": _data_uri(result["figure_2d"]),
            },
        ]
        surface = np.array(result["accuracy_surface"], dtype=float)
        summary = {
            "average_accuracy": round(float(surface.mean()), 4),
            "peak_accuracy": round(float(surface.max()), 4),
            "final_step_accuracy": round(float(surface[:, -1].mean()), 4),
            "decay_range": [round(float(decay_rates[0]), 4), round(float(decay_rates[-1]), 4)],
            "time_steps": int(result["time_steps"]),
        }
        title = "Simulation 1 - Recommendation Accuracy"

    elif simulation == "2":
        decay_rates = np.linspace(
            _safe_float(payload.get("decay_start"), 0.03, minimum=0.0),
            _safe_float(payload.get("decay_end"), 0.4, minimum=0.0),
            _safe_int(payload.get("decay_points"), 8, minimum=2, maximum=50),
        )
        engagement_levels = np.linspace(
            _safe_float(payload.get("engagement_start"), 0.2, minimum=0.0),
            _safe_float(payload.get("engagement_end"), 0.8, minimum=0.0),
            _safe_int(payload.get("engagement_points"), 8, minimum=2, maximum=50),
        )
        result = simulation_2_privacy_engagement_tradeoff(
            n_users=_safe_int(payload.get("n_users"), 18, minimum=1, maximum=40),
            n_items=_safe_int(payload.get("n_items"), 16, minimum=2, maximum=50),
            feature_dim=_safe_int(payload.get("feature_dim"), 6, minimum=2, maximum=20),
            decay_rates=decay_rates,
            engagement_levels=engagement_levels,
            random_seed=_safe_int(payload.get("random_seed"), 42, minimum=0, maximum=9999),
        )
        figures = [
            {
                "filename": Path(result["figure_3d"]).name,
                "label": "Simulation 2 - Privacy vs Engagement Surface",
                "static_url": f"/screenshots/{Path(result['figure_3d']).name}",
                "data_uri": _data_uri(result["figure_3d"]),
            },
            {
                "filename": Path(result["figure_contour"]).name,
                "label": "Simulation 2 - Contour Comparison",
                "static_url": f"/screenshots/{Path(result['figure_contour']).name}",
                "data_uri": _data_uri(result["figure_contour"]),
            },
        ]
        privacy = np.array(result["privacy_surface"], dtype=float)
        engagement = np.array(result["engagement_surface"], dtype=float)
        summary = {
            "average_privacy_exposure": round(float(privacy.mean()), 4),
            "average_engagement": round(float(engagement.mean()), 4),
            "lowest_privacy_exposure": round(float(privacy.min()), 4),
            "highest_engagement": round(float(engagement.max()), 4),
        }
        title = "Simulation 2 - Privacy vs Engagement"

    elif simulation == "3":
        result = simulation_3_comparative_analysis(
            n_users=_safe_int(payload.get("n_users"), 16, minimum=1, maximum=40),
            n_items=_safe_int(payload.get("n_items"), 18, minimum=2, maximum=60),
            feature_dim=_safe_int(payload.get("feature_dim"), 8, minimum=2, maximum=20),
            time_steps=_safe_int(payload.get("time_steps"), 12, minimum=2, maximum=30),
            random_seed=_safe_int(payload.get("random_seed"), 42, minimum=0, maximum=9999),
        )
        figures = [
            {
                "filename": Path(result["figure"]).name,
                "label": "Simulation 3 - Comparative Performance",
                "static_url": f"/screenshots/{Path(result['figure']).name}",
                "data_uri": _data_uri(result["figure"]),
            },
            {
                "filename": Path(result["table"]).name,
                "label": "Simulation 3 - Comparative Table",
                "static_url": f"/screenshots/{Path(result['table']).name}",
                "data_uri": _data_uri(result["table"]),
            },
        ]
        summary = {"final_scores": result["final_scores"]}
        title = "Simulation 3 - Comparative Analysis"

    elif simulation == "4":
        result = simulation_4_entropy_and_dynamics(
            n_users=_safe_int(payload.get("n_users"), 18, minimum=1, maximum=40),
            time_steps=_safe_int(payload.get("time_steps"), 24, minimum=2, maximum=60),
        )
        figures = [
            {
                "filename": Path(result["figure"]).name,
                "label": "Simulation 4 - Dynamics Over Time",
                "static_url": f"/screenshots/{Path(result['figure']).name}",
                "data_uri": _data_uri(result["figure"]),
            }
        ]
        summary = {}
        for label in result["entropy"]:
            entropy = np.array(result["entropy"][label], dtype=float)
            exposure = np.array(result["exposure_risk"][label], dtype=float)
            decay = np.array(result["decay_coefficient"][label], dtype=float)
            summary[label] = {
                "entropy_mean": round(float(entropy.mean()), 4) if entropy.size else 0.0,
                "exposure_mean": round(float(exposure.mean()), 4) if exposure.size else 0.0,
                "decay_mean": round(float(decay.mean()), 4) if decay.size else 0.0,
            }
        title = "Simulation 4 - Entropy and Dynamics"

    else:
        raise ValueError("Unknown simulation selection")

    return {
        "simulation": f"simulation_{simulation}",
        "title": title,
        "summary": summary,
        "figures": figures,
        "run_id": str(np.int64(np.random.randint(1, 10**9))),
    }


def app(environ, start_response) -> Iterable[bytes]:
    path = environ.get("PATH_INFO", "/")

    if path == "/run":
        if environ.get("REQUEST_METHOD", "GET").upper() != "POST":
            return _json_response(start_response, "405 Method Not Allowed", {"ok": False, "error": "Use POST for /run"})

        try:
            payload = json.loads(_read_request_body(environ).decode("utf-8") or "{}")
            result = _run_simulation(payload)
            result["ok"] = True
            return _json_response(start_response, "200 OK", result)
        except Exception as exc:  # pragma: no cover - surfaced in UI
            return _json_response(start_response, "400 Bad Request", {"ok": False, "error": str(exc)})

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
            ("Cache-Control", "no-store"),
        ]
        start_response("200 OK", headers)
        return [body]

    if path.startswith("/screenshots/"):
        relative = Path(path.removeprefix("/screenshots/")).name
        candidate = SCREENSHOTS_DIR / relative
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
