"""Build-time JIT warmup via forge-middleware (shares forge-middleware.toml)."""

from forge_middleware import Middleware
from forge_middleware.config import load_config

cfg = load_config("forge-middleware.toml")
mw = Middleware.from_config(cfg)
report = mw.warmup(timeout_s=cfg.settings.timeout_s)

bad = {k: v for k, v in report.items() if v["status"] in ("failed", "timeout")}
if bad:
    raise SystemExit(f"build-time warmup failed: {bad}")
print(f"[warmup_build] OK — {mw.stats()}")
