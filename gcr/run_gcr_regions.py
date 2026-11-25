#!/usr/bin/env python3
"""Comparação de métricas entre regiões do Cloud Run (yuki-mt e fbosler).

Reusa a medição de ``run_gcr_requests.py`` e a repete para uma lista de regiões.
A imagem é a mesma em todas as regiões, de modo que a diferença observada vem da
região, e não de um novo build. Com ``--deploy`` as mesmas imagens são
implantadas em cada região (sem rebuild), antes da medição.

Uso:
    python benchmarks/gcr/run_gcr_regions.py --project tcc-dev-496317 \
        --regions us-central1 europe-west1 southamerica-east1 --requests 30

    # implantando as mesmas imagens (gcr.io, sem rebuild) em cada região
    python benchmarks/gcr/run_gcr_regions.py --project tcc-dev-496317 \
        --regions us-central1 europe-west1 --deploy

Notas de medição:
  - wall_ms inclui a rede entre a máquina que mede e a região. Para comparar
    compute use server_ms/billed_ms; para comparar latência percebida, o local do
    cliente faz parte do resultado.
  - O custo usa os preços de referência Tier 1 us-central1. Regiões de Tier 2
    custam diferente; ajuste os preços no módulo base se comparar Tier 1 com Tier 2.
"""

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_gcr_requests import (  # noqa: E402
    APPS,
    DEFAULT_BASE_URL,
    measure,
    now_utc,
    resolve_resources,
    resolve_url,
)

REGION_APPS = ["yuki-mt", "fbosler"]


def deploy_service(project, region, svc, image, cpu, mem_gb, min_instances):
    """Implanta a imagem já construída em uma região, sem rebuild."""
    cmd = [
        "gcloud", "run", "deploy", svc,
        "--project", project,
        "--region", region,
        "--image", image,
        "--cpu", str(cpu),
        "--memory", f"{mem_gb}Gi",
        "--min-instances", str(min_instances),
        "--scaling", "auto",
        "--allow-unauthenticated",
        "--quiet",
    ]
    print("    $ " + " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def measure_variant(app, variant, project, region, requests, warmups, cpu, mem_gb):
    cfg = APPS[app]
    svc = f"{cfg['svc']}-{variant}"
    url = resolve_url(svc, cfg["path"], project, region, DEFAULT_BASE_URL)
    r_cpu, r_mem = resolve_resources(svc, project, region, cpu, mem_gb)
    print(f"  [{app}/{variant}] {url} (cpu={r_cpu}, mem={r_mem}GiB)", flush=True)
    result = measure(url, cfg["payload"], requests, warmups, r_cpu, r_mem)
    result["url"] = url
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", default="tcc-dev-496317")
    parser.add_argument("--regions", nargs="+", required=True)
    parser.add_argument("--apps", nargs="*", default=REGION_APPS)
    parser.add_argument("--variants", nargs="*", default=["original", "optimized"])
    parser.add_argument("--requests", type=int, default=30)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--deploy", action="store_true",
                        help="implanta as mesmas imagens em cada região antes de medir")
    parser.add_argument("--image-template", default="gcr.io/tcc-dev-496317/{svc}:latest",
                        help="referência da imagem por serviço; {svc} vira <app>-<variante>")
    parser.add_argument("--cpu", type=float, default=1.0)
    parser.add_argument("--memory-gb", type=float, default=0.5)
    parser.add_argument("--min-instances", type=int, default=0)
    parser.add_argument(
        "--output",
        default=str(Path(__file__).parent / f"gcr_regions_{time.strftime('%Y%m%d_%H%M')}.json"),
    )
    args = parser.parse_args()

    results = {
        "metadata": {
            "start_time": now_utc(),
            "project": args.project,
            "regions": args.regions,
            "apps": args.apps,
            "variants": args.variants,
            "requests_per_service": args.requests,
            "warmups_per_service": args.warmups,
            "note": "mesma imagem em todas as regiões; custo em preços Tier 1 us-central1",
        },
        "regions": {},
    }

    for region in args.regions:
        print(f"\n{'=' * 70}\n# região {region}\n{'=' * 70}", flush=True)
        region_result = {"apps": {}}
        for app in args.apps:
            app_result = {}
            for variant in args.variants:
                svc = f"{APPS[app]['svc']}-{variant}"
                try:
                    if args.deploy:
                        deploy_service(
                            args.project, region, svc,
                            args.image_template.format(svc=svc),
                            args.cpu, args.memory_gb, args.min_instances,
                        )
                    app_result[variant] = measure_variant(
                        app, variant, args.project, region,
                        args.requests, args.warmups, args.cpu, args.memory_gb,
                    )
                except Exception as exc:  # noqa: BLE001
                    print(f"    FALHA [{app}/{variant}]: {type(exc).__name__}: {exc}", flush=True)
                    app_result[variant] = {"error": f"{type(exc).__name__}: {exc}"}
            region_result["apps"][app] = app_result
        results["regions"][region] = region_result

    results["metadata"]["end_time"] = now_utc()
    Path(args.output).write_text(json.dumps(results, indent=2, ensure_ascii=False))

    print(f"\n{'=' * 100}")
    print("RESUMO — médias por região (mesma imagem, sem rebuild)")
    print(f"{'=' * 100}")
    header = f"{'app':<14}{'variante':<11}{'região':<22}{'server_ms':>11}{'billed_ms':>11}{'wall_ms':>10}{'custo/1M':>10}"
    print(header)
    print("-" * 100)
    for app in args.apps:
        for variant in args.variants:
            for region in args.regions:
                r = results["regions"][region]["apps"][app].get(variant, {})
                if "error" in r or r.get("server_ms") is None:
                    print(f"{app:<14}{variant:<11}{region:<22}{'FALHA':>11}")
                    continue
                s = r.get("server_ms", {})
                b = r.get("billed_ms", {})
                w = r.get("wall_ms", {})
                c = r.get("cost_usd", {})
                print(
                    f"{app:<14}{variant:<11}{region:<22}"
                    f"{s.get('mean', 0):>11.1f}{b.get('mean', 0):>11.1f}"
                    f"{w.get('mean', 0):>10.1f}{('$%.2f' % (c.get('mean', 0) * 1e6)):>10}"
                )
            print("-" * 100)

    print(f"\nResultados salvos em: {args.output}")


if __name__ == "__main__":
    main()
