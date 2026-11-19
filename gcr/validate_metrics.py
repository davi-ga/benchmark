#!/usr/bin/env python3
"""
Valida as métricas de cobrança do Cloud Run contra a medição local.

Consulta, POR REVISÃO (`resource.labels.revision_name`):
  - run.googleapis.com/container/billable_instance_time (ALIGN_SUM)
  - run.googleapis.com/request_count (ALIGN_SUM e ALIGN_RATE)

e compara com o que foi medido localmente no JSON do run_gcr_requests.py.

Uso:
    python benchmarks/gcr/validate_metrics.py \
        --project tcc-dev-496317 \
        --results /tmp/gcr_all_cost_v2.json
    python benchmarks/gcr/validate_metrics.py --project tcc-dev-496317 \
        --results /tmp/gcr_all_cost_v2.json --label service
"""

import argparse
import json
import subprocess
import urllib.parse
import urllib.request
from datetime import datetime, timezone, timedelta

from run_gcr_requests import APPS

DEFAULT_RESULTS = "/tmp/gcr_all_cost_v2.json"
PERIOD_S = 60


def rfc3339(dt: datetime) -> str:
    return dt.strftime("%Y-%m-%dT%H:%M:%SZ")


def parse_rfc3339(value: str) -> datetime:
    return datetime.strptime(value, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)


def access_token() -> str:
    return subprocess.check_output(["gcloud", "auth", "print-access-token"], text=True).strip()


def gcloud_json(args: list[str]) -> dict | None:
    proc = subprocess.run(args, capture_output=True, text=True)
    try:
        return json.loads(proc.stdout)
    except (json.JSONDecodeError, TypeError):
        return None


def latest_revision(svc: str, project: str, region: str) -> str | None:
    data = gcloud_json(
        ["gcloud", "run", "services", "describe", svc, "--project", project,
         "--region", region, "--format", "json"]
    )
    return (data or {}).get("status", {}).get("latestReadyRevisionName")


def query(
    project: str,
    metric: str,
    label_key: str,
    label_value: str,
    start: str,
    end: str,
    aligner: str,
    token: str,
) -> dict:
    flt = (
        f'metric.type="{metric}" AND resource.type="cloud_run_revision" '
        f'AND resource.labels.{label_key}="{label_value}"'
    )
    qs = urllib.parse.urlencode({
        "filter": flt,
        "interval.startTime": start,
        "interval.endTime": end,
        "aggregation.alignmentPeriod": f"{PERIOD_S}s",
        "aggregation.perSeriesAligner": aligner,
        "pageSize": "200",
    })
    url = f"https://monitoring.googleapis.com/v3/projects/{project}/timeSeries?{qs}"
    req = urllib.request.Request(url, headers={"Authorization": f"Bearer {token}"})
    try:
        data = json.load(urllib.request.urlopen(req, timeout=30))
    except Exception as exc:  # noqa: BLE001
        return {"error": str(exc)[:200], "total": 0.0, "points": 0, "kind": None}

    total = 0.0
    points = 0
    kind = None
    for ts in data.get("timeSeries", []):
        kind = ts.get("metricKind")
        for p in ts.get("points", []):
            value = p.get("value", {})
            total += float(value.get("doubleValue", value.get("int64Value", 0)))
            points += 1
    return {"total": total, "points": points, "kind": kind}


def expected_from_results(results: dict, app: str, variant: str) -> tuple[float, int]:
    """(billed seconds, request count) measured locally, cold + warm."""
    measured = results["apps"][app][variant]
    total_s = 0.0
    n = 0
    cold = measured.get("cold")
    for run in ([cold] if cold else []) + measured.get("runs", []):
        if not run or not run.get("ok"):
            continue
        duration = run.get("billed_ms")
        if duration is None:
            duration = run.get("server_ms")
        if duration is not None:
            total_s += duration / 1000.0
            n += 1
    return total_s, n


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", required=True)
    parser.add_argument("--region", default="us-central1")
    parser.add_argument("--results", default=DEFAULT_RESULTS)
    parser.add_argument("--label", choices=["revision", "service"], default="revision")
    parser.add_argument("--window-start", default=None, help="RFC3339; default = results start")
    parser.add_argument("--window-end", default=None, help="RFC3339; default = now + 5 min")
    args = parser.parse_args()

    results = json.load(open(args.results))
    meta = results.get("metadata", {})
    start = args.window_start or meta.get("start_time")
    end = args.window_end or rfc3339(
        max(datetime.now(timezone.utc), parse_rfc3339(meta.get("end_time", rfc3339(datetime.now(timezone.utc)))))
        + timedelta(minutes=5)
    )
    label_key = "revision_name" if args.label == "revision" else "service_name"
    token = access_token()

    print(f"janela: {start} .. {end} | label={args.label}\n")
    header = (
        f"{'app':<14}{'variant':<11}{'rev':<10}{'billable_s':>12}{'pts':>4}"
        f"{'req_sum':>9}{'sum_pts':>8}{'req_rate*60':>12}{'rate_pts':>9}"
        f"{'esperado_s':>12}{'esperado_n':>11}"
    )
    print(header)
    print("-" * len(header))

    for app, variants in results["apps"].items():
        for variant in meta.get("variants", ["original", "optimized"]):
            measured = variants.get(variant, {})
            svc = f"{APPS[app]['svc']}-{variant}"

            label_value = latest_revision(svc, args.project, args.region) if args.label == "revision" else svc
            if not label_value:
                print(f"{app:<14}{variant:<11}{'?':<10}  (revisão/serviço não encontrado)")
                continue

            bill = query(args.project, "run.googleapis.com/container/billable_instance_time",
                         label_key, label_value, start, end, "ALIGN_SUM", token)
            req_sum = query(args.project, "run.googleapis.com/request_count",
                            label_key, label_value, start, end, "ALIGN_SUM", token)
            req_rate = query(args.project, "run.googleapis.com/request_count",
                             label_key, label_value, start, end, "ALIGN_RATE", token)

            expected_s, expected_n = expected_from_results(results, app, variant)
            rev_short = label_value[-9:] if args.label == "revision" else label_value
            print(
                f"{app:<14}{variant:<11}{rev_short:<10}{bill['total']:>12.2f}{bill['points']:>4}"
                f"{req_sum['total']:>9.0f}{req_sum['points']:>8}"
                f"{req_rate['total'] * PERIOD_S:>12.1f}{req_rate['points']:>9}"
                f"{expected_s:>12.2f}{expected_n:>11}"
            )


if __name__ == "__main__":
    main()
