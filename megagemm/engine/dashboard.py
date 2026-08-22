"""
📊 Dashboard — Live Monitoring Dashboard for MegaGemm
------------------------------------------------------
Self-contained HTTP server with embedded HTML/JS dashboard.
Shows real-time inference metrics, GPU stats, and XAI quality.

Runs in a background thread — zero interference with inference.

Usage:
    from megagemm.engine.dashboard import DashboardServer

    server = DashboardServer(monitor, port=8080)
    server.start()    # Non-blocking — runs in background thread
    # ... do inference ...
    server.stop()

Author: Gabriel Yogi
"""

import json
import threading
from http.server import HTTPServer, BaseHTTPRequestHandler
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from .monitor import InferenceMonitor


__all__ = ['DashboardServer']


# ─────────────────────────────────────────────────────────────────────────────
# Embedded HTML Dashboard (self-contained, no external dependencies)
# ─────────────────────────────────────────────────────────────────────────────

DASHBOARD_HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>MegaGemm Live Dashboard</title>
<style>
  :root {
    --bg: #0f0f17; --card: #1a1a2e; --border: #2a2a4a;
    --text: #e0e0f0; --dim: #8888aa; --accent: #6c5ce7;
    --green: #00b894; --yellow: #fdcb6e; --red: #e17055;
    --blue: #74b9ff;
  }
  * { margin: 0; padding: 0; box-sizing: border-box; }
  body {
    font-family: 'Segoe UI', system-ui, -apple-system, sans-serif;
    background: var(--bg); color: var(--text);
    min-height: 100vh; padding: 20px;
  }
  .header {
    text-align: center; margin-bottom: 24px;
    padding: 16px; border-bottom: 1px solid var(--border);
  }
  .header h1 { font-size: 1.8em; font-weight: 700; }
  .header h1 span { color: var(--accent); }
  .header .meta { color: var(--dim); font-size: 0.85em; margin-top: 4px; }
  .status-dot {
    display: inline-block; width: 8px; height: 8px;
    border-radius: 50%; background: var(--green); margin-right: 6px;
    animation: pulse 2s ease-in-out infinite;
  }
  @keyframes pulse {
    0%, 100% { opacity: 1; } 50% { opacity: 0.4; }
  }
  .grid {
    display: grid; gap: 16px; max-width: 1200px; margin: 0 auto;
    grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
  }
  .card {
    background: var(--card); border: 1px solid var(--border);
    border-radius: 12px; padding: 20px;
    transition: border-color 0.3s;
  }
  .card:hover { border-color: var(--accent); }
  .card h3 {
    font-size: 0.8em; text-transform: uppercase; letter-spacing: 1px;
    color: var(--dim); margin-bottom: 12px;
  }
  .metric {
    display: flex; justify-content: space-between;
    align-items: baseline; margin-bottom: 8px;
  }
  .metric .label { color: var(--dim); font-size: 0.9em; }
  .metric .value { font-size: 1.1em; font-weight: 600; }
  .big-metric .value { font-size: 2em; font-weight: 700; color: var(--accent); }
  .bar-container {
    background: #2a2a3e; border-radius: 6px; height: 8px;
    margin-top: 6px; overflow: hidden;
  }
  .bar-fill {
    height: 100%; border-radius: 6px; transition: width 0.6s ease;
  }
  .bar-green { background: linear-gradient(90deg, var(--green), #55efc4); }
  .bar-yellow { background: linear-gradient(90deg, var(--yellow), #ffeaa7); }
  .bar-red { background: linear-gradient(90deg, var(--red), #fab1a0); }
  .bar-blue { background: linear-gradient(90deg, var(--accent), var(--blue)); }
  .risk-badge {
    display: inline-block; padding: 2px 10px; border-radius: 12px;
    font-size: 0.8em; font-weight: 600;
  }
  .risk-LOW { background: rgba(0,184,148,0.2); color: var(--green); }
  .risk-MEDIUM { background: rgba(253,203,110,0.2); color: var(--yellow); }
  .risk-HIGH { background: rgba(225,112,85,0.2); color: var(--red); }
  .recent-list { max-height: 200px; overflow-y: auto; }
  .recent-item {
    padding: 8px 12px; margin-bottom: 4px; border-radius: 8px;
    background: rgba(255,255,255,0.03); font-size: 0.85em;
    display: flex; justify-content: space-between; align-items: center;
  }
  .recent-item .prompt {
    overflow: hidden; text-overflow: ellipsis; white-space: nowrap;
    max-width: 60%; color: var(--dim);
  }
  .waiting { text-align: center; padding: 40px; color: var(--dim); }
  .footer {
    text-align: center; margin-top: 24px; padding: 12px;
    color: var(--dim); font-size: 0.75em; border-top: 1px solid var(--border);
  }
</style>
</head>
<body>
<div class="header">
  <h1><span>MegaGemm</span> Live Dashboard</h1>
  <div class="meta"><span class="status-dot"></span>Auto-refresh every 2s | <span id="uptime">-</span></div>
</div>

<div class="grid" id="dashboard">
  <div class="waiting">Waiting for data...</div>
</div>

<div class="footer">MegaGemm Inference Engine &mdash; XAI + Observability</div>

<script>
function fmt(n, d=1) { return typeof n === 'number' ? n.toFixed(d) : n || '-'; }
function fmtMs(ms) { return ms > 1000 ? (ms/1000).toFixed(1)+'s' : ms.toFixed(0)+'ms'; }
function fmtUptime(s) {
  if (s < 60) return s.toFixed(0) + 's';
  if (s < 3600) return Math.floor(s/60) + 'm ' + (s%60).toFixed(0) + 's';
  return Math.floor(s/3600) + 'h ' + Math.floor((s%3600)/60) + 'm';
}
function barClass(pct) { return pct > 80 ? 'bar-red' : pct > 50 ? 'bar-yellow' : 'bar-green'; }

function renderDashboard(stats, recent) {
  const p = stats.performance || {};
  const q = stats.quality || {};
  const t = stats.throughput || {};
  const g = stats.gpu || {};

  document.getElementById('uptime').textContent = fmtUptime(stats.uptime_seconds || 0);

  let html = '';

  // Overview card
  html += `<div class="card">
    <h3>Overview</h3>
    <div class="metric big-metric"><span class="label">Requests</span>
      <span class="value">${t.total_requests || 0}</span></div>
    <div class="metric"><span class="label">RPS</span>
      <span class="value">${fmt(stats.requests_per_second, 3)}</span></div>
    <div class="metric"><span class="label">Tokens Generated</span>
      <span class="value">${t.total_tokens_out || 0}</span></div>
  </div>`;

  // Performance card
  html += `<div class="card">
    <h3>Performance</h3>
    <div class="metric"><span class="label">Avg Latency</span>
      <span class="value">${fmtMs(p.avg_latency_ms||0)}</span></div>
    <div class="metric"><span class="label">P95 Latency</span>
      <span class="value">${fmtMs(p.p95_latency_ms||0)}</span></div>
    <div class="metric"><span class="label">TTFT</span>
      <span class="value">${fmtMs(p.avg_ttft_ms||0)}</span></div>
    <div class="metric"><span class="label">Speed</span>
      <span class="value">${fmt(p.avg_tps)} tok/s</span></div>
    <div class="metric"><span class="label">Prefill</span>
      <span class="value">${fmtMs(p.avg_prefill_ms||0)}</span></div>
    <div class="metric"><span class="label">Decode</span>
      <span class="value">${fmtMs(p.avg_decode_ms||0)}</span></div>
  </div>`;

  // GPU card
  if (g.gpu_name) {
    const vramPct = g.vram_usage_pct || 0;
    html += `<div class="card">
      <h3>GPU Resources</h3>
      <div class="metric"><span class="label">Device</span>
        <span class="value">${g.gpu_name}</span></div>
      <div class="metric"><span class="label">VRAM Used</span>
        <span class="value">${fmt(g.vram_used_mb,0)} / ${fmt(g.vram_total_mb,0)} MB</span></div>
      <div class="bar-container">
        <div class="bar-fill ${barClass(vramPct)}" style="width:${vramPct}%"></div>
      </div>
      <div class="metric" style="margin-top:8px"><span class="label">Usage</span>
        <span class="value">${fmt(vramPct)}%</span></div>
      <div class="metric"><span class="label">Free</span>
        <span class="value">${fmt(g.vram_free_mb,0)} MB</span></div>
    </div>`;
  }

  // Quality card (XAI)
  if (q.xai_enabled_requests > 0) {
    const hr = q.hallucination_rate_high || 0;
    const riskClass = hr === 0 ? 'LOW' : (hr < 0.2 ? 'MEDIUM' : 'HIGH');
    const rd = q.risk_distribution || {};
    const confPct = (q.avg_confidence || 0) * 100;

    html += `<div class="card">
      <h3>Quality (XAI)</h3>
      <div class="metric"><span class="label">Confidence</span>
        <span class="value">${fmt(q.avg_confidence, 4)}</span></div>
      <div class="bar-container">
        <div class="bar-fill bar-blue" style="width:${confPct}%"></div>
      </div>
      <div class="metric" style="margin-top:8px"><span class="label">Entropy</span>
        <span class="value">${fmt(q.avg_entropy, 4)}</span></div>
      <div class="metric"><span class="label">Halluc. Rate</span>
        <span class="value"><span class="risk-badge risk-${riskClass}">${(hr*100).toFixed(1)}% HIGH</span></span></div>
      <div class="metric"><span class="label">Distribution</span>
        <span class="value" style="font-size:0.85em">
          <span style="color:var(--green)">${rd.LOW||0}</span> /
          <span style="color:var(--yellow)">${rd.MEDIUM||0}</span> /
          <span style="color:var(--red)">${rd.HIGH||0}</span>
        </span>
      </div>
    </div>`;
  }

  // Throughput card
  html += `<div class="card">
    <h3>Throughput</h3>
    <div class="metric"><span class="label">Tokens In</span>
      <span class="value">${t.total_tokens_in || 0}</span></div>
    <div class="metric"><span class="label">Tokens Out</span>
      <span class="value">${t.total_tokens_out || 0}</span></div>
    <div class="metric"><span class="label">Avg In/req</span>
      <span class="value">${fmt(t.avg_tokens_in, 0)}</span></div>
    <div class="metric"><span class="label">Avg Out/req</span>
      <span class="value">${fmt(t.avg_tokens_out, 0)}</span></div>
  </div>`;

  // Recent requests card
  if (recent && recent.length > 0) {
    let items = recent.map(r => {
      const risk = r.hallucination_risk || '';
      const badge = risk ? `<span class="risk-badge risk-${risk}">${risk}</span>` : '';
      return `<div class="recent-item">
        <span class="prompt">${r.prompt}</span>
        ${badge}
      </div>`;
    }).join('');

    html += `<div class="card" style="grid-column: span 2">
      <h3>Recent Requests</h3>
      <div class="recent-list">${items}</div>
    </div>`;
  }

  document.getElementById('dashboard').innerHTML = html;
}

async function refresh() {
  try {
    const [statsRes, recentRes] = await Promise.all([
      fetch('/api/stats'), fetch('/api/recent')
    ]);
    const stats = await statsRes.json();
    const recent = await recentRes.json();
    renderDashboard(stats, recent);
  } catch(e) { console.error('Refresh failed:', e); }
}

refresh();
setInterval(refresh, 2000);
</script>
</body>
</html>"""


# ─────────────────────────────────────────────────────────────────────────────
# HTTP Server (stdlib only — no Flask/FastAPI needed)
# ─────────────────────────────────────────────────────────────────────────────

class _DashboardHandler(BaseHTTPRequestHandler):
    """HTTP handler that serves the dashboard and API endpoints."""

    monitor: Optional['InferenceMonitor'] = None  # Set by DashboardServer

    def do_GET(self):
        if self.path == '/' or self.path == '/dashboard':
            self._send_html(DASHBOARD_HTML)
        elif self.path == '/api/stats':
            self._send_json(self.monitor.get_stats() if self.monitor else {})
        elif self.path == '/api/recent':
            self._send_json(self.monitor.get_recent(20) if self.monitor else [])
        else:
            self.send_error(404)

    def _send_html(self, content: str):
        self.send_response(200)
        self.send_header('Content-Type', 'text/html; charset=utf-8')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()
        self.wfile.write(content.encode('utf-8'))

    def _send_json(self, data):
        self.send_response(200)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()
        self.wfile.write(json.dumps(data, ensure_ascii=False).encode('utf-8'))

    def log_message(self, format, *args):
        """Suppress default access logging to keep console clean."""
        pass


class _ReusableHTTPServer(HTTPServer):
    """HTTPServer with SO_REUSEADDR enabled (fixes 'Address already in use')."""
    allow_reuse_address = True


class DashboardServer:
    """
    Self-contained live monitoring dashboard.

    Runs an HTTP server in a background daemon thread.
    Serves an embedded HTML page with auto-refreshing metrics.

    Endpoints:
        GET /           → Dashboard HTML page
        GET /api/stats  → JSON monitoring stats
        GET /api/recent → JSON recent requests
    """

    def __init__(self, monitor: 'InferenceMonitor', port: int = 8080):
        self._monitor = monitor
        self._port = port
        self._server: Optional[HTTPServer] = None
        self._thread: Optional[threading.Thread] = None

    def start(self) -> None:
        """Start the dashboard server in a background thread."""
        # Create handler class with monitor reference
        handler = type('Handler', (_DashboardHandler,), {'monitor': self._monitor})

        # Try to bind — handles "Address already in use" by trying next ports
        for port in [self._port, self._port + 1, self._port + 2]:
            try:
                server = _ReusableHTTPServer(('0.0.0.0', port), handler)
                self._server = server
                self._port = port
                break
            except OSError:
                continue
        else:
            print(f"⚠️  Dashboard: Could not bind to ports {self._port}-{self._port+2}")
            return

        self._thread = threading.Thread(
            target=self._server.serve_forever,
            daemon=True,
            name='megagemm-dashboard',
        )
        self._thread.start()
        print(f"📊 Dashboard running at http://localhost:{self._port}")

        # Auto-display in Colab/Kaggle notebooks
        if self._is_colab():
            self._show_colab_iframe()
        elif self._is_kaggle():
            self._show_kaggle_iframe()

    def _is_colab(self) -> bool:
        """Check if running in Google Colab."""
        try:
            import google.colab  # noqa
            return True
        except ImportError:
            return False

    def _is_kaggle(self) -> bool:
        """Check if running in Kaggle notebook."""
        import os
        return os.path.exists('/kaggle/working')

    def _show_colab_iframe(self) -> None:
        """Show dashboard inline in Colab using port proxy."""
        try:
            from google.colab.output import serve_kernel_port_as_iframe
            print(f"🌐 Rendering dashboard inline in Colab...")
            serve_kernel_port_as_iframe(self._port, height=600)
        except Exception as e:
            print(f"⚠️  Could not render inline: {e}")
            print(f"   Try running in a cell: from google.colab.output import serve_kernel_port_as_iframe; serve_kernel_port_as_iframe({self._port})")

    def _show_kaggle_iframe(self) -> None:
        """Show dashboard inline in Kaggle using IPython IFrame."""
        try:
            from IPython.display import display, IFrame
            print(f"🌐 Rendering dashboard inline in Kaggle...")
            display(IFrame(src=f"http://localhost:{self._port}", width="100%", height=600))
        except Exception as e:
            print(f"⚠️  Could not render inline: {e}")

    def show_inline(self, height: int = 600) -> None:
        """
        Manually render dashboard inline in a notebook.

        Call this in a new cell if the auto-display didn't work.
        Works in Colab, Kaggle, and Jupyter notebooks.
        """
        if not self.is_running:
            print("⚠️  Dashboard not running. Call start() first.")
            return

        if self._is_colab():
            self._show_colab_iframe()
        else:
            try:
                from IPython.display import display, IFrame
                display(IFrame(src=f"http://localhost:{self._port}", width="100%", height=height))
            except ImportError:
                print(f"📊 Open in browser: http://localhost:{self._port}")

    def stop(self) -> None:
        """Stop the dashboard server."""
        if self._server:
            self._server.shutdown()
            self._server = None
            self._thread = None
            print("📊 Dashboard stopped")

    @property
    def url(self) -> str:
        return f"http://localhost:{self._port}"

    @property
    def is_running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()
