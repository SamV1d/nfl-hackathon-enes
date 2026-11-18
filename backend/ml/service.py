"""Caché perezosa de la auditoría: se calcula en la primera consulta y se reutiliza."""
from __future__ import annotations

import logging
import threading

log = logging.getLogger("backend.ml")


class AuditUnavailable(Exception):
    """La auditoría no se pudo calcular con los datos cargados."""


class AuditCache:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._result: dict | None = None
        self._error: str | None = None

    def get(self, compute) -> dict:
        """`compute` es un callable sin argumentos que devuelve el dict de la auditoría."""
        with self._lock:
            if self._result is None and self._error is None:
                try:
                    self._result = compute()
                except Exception as exc:  # noqa: BLE001 - se informa como no disponible
                    log.exception("fallo la auditoría de modelos")
                    self._error = type(exc).__name__
            if self._result is None:
                raise AuditUnavailable(self._error or "unknown")
            return self._result
