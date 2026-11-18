"""Explicación opcional con Amazon Nova (Bedrock). Ningún endpoint central depende de este módulo.

El cliente se crea solo con la región. No se leen, verifican ni registran credenciales.
"""
from __future__ import annotations

import logging
import os
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeout
from dataclasses import dataclass
from typing import Any, Callable, Mapping

log = logging.getLogger(__name__)

DEFAULT_REGION = "us-east-1"
DEFAULT_MODEL_ID = "us.amazon.nova-pro-v1:0"
TIMEOUT_SECONDS = 10.0
MAX_CHARS = 1000
UNAVAILABLE_MESSAGE = "explicación no disponible"


@dataclass(frozen=True)
class Explanation:
    available: bool
    text: str | None = None
    message: str | None = None


def _unavailable() -> Explanation:
    return Explanation(available=False, message=UNAVAILABLE_MESSAGE)


def build_prompt(rec: Mapping[str, Any]) -> str:
    """Prompt con la mejor llamada, su probabilidad, sus yardas y la diferencia vs la típica."""
    best = rec["ranked"][0]
    call = best["call"]
    parts = [
        "Eres asistente de un head coach de la NFL. Explica en español, en lenguaje llano y en "
        f"menos de {MAX_CHARS} caracteres, por qué se recomienda esta jugada defensiva.",
        f"Cobertura: {call['coverage']}. Rushers: {call['pass_rushers']}. "
        f"Defensores en la caja: {call['defenders_in_box']}.",
        f"Probabilidad de presión: {float(best['pressure_probability']) * 100:.1f}%.",
        f"Yardas esperadas: {float(best['expected_yards']):.1f}.",
    ]
    delta = rec.get("yards_delta_vs_typical")
    if delta is not None:
        parts.append(f"Diferencia de yardas contra la llamada típica: {float(delta):+.1f}.")
    return "\n".join(parts)


def _invoke(prompt: str, region: str, model_id: str) -> str:
    import boto3  # import diferido: el arranque no depende de boto3

    client = boto3.client("bedrock-runtime", region_name=region)
    resp = client.converse(
        modelId=model_id,
        messages=[{"role": "user", "content": [{"text": prompt}]}],
        inferenceConfig={"maxTokens": 400, "temperature": 0.2},
    )
    return resp["output"]["message"]["content"][0]["text"]


def explain(
    rec: Mapping[str, Any],
    invoke: Callable[[str, str, str], str] = _invoke,
    timeout: float = TIMEOUT_SECONDS,
) -> Explanation:
    """Devuelve la explicación (<= 1.000 caracteres) o `Explanation(available=False)`.

    Cualquier fallo (prompt, Bedrock, vacío, timeout) se traduce en "no disponible".
    """
    # rules.md: región y modelo con valor por defecto; un valor vacío cuenta como ausente.
    region = os.environ.get("AWS_REGION") or DEFAULT_REGION
    model_id = os.environ.get("BEDROCK_MODEL_ID") or DEFAULT_MODEL_ID

    pool = ThreadPoolExecutor(max_workers=1)
    try:
        prompt = build_prompt(rec)
        text = pool.submit(invoke, prompt, region, model_id).result(timeout=timeout)
    except FutureTimeout:
        log.warning("Nova: timeout tras %.1f s", timeout)
        return _unavailable()
    except Exception as exc:  # noqa: BLE001 - solo se registra el tipo, nunca el detalle
        log.warning("Nova: fallo (%s)", type(exc).__name__)
        return _unavailable()
    finally:
        pool.shutdown(wait=False, cancel_futures=True)

    if not isinstance(text, str) or not text.strip():
        return _unavailable()
    return Explanation(available=True, text=text.strip()[:MAX_CHARS])
