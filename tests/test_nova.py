"""Propiedad 25 y pruebas unitarias de Nova con Bedrock simulado."""
import sys
import time
import types
from collections.abc import Mapping

import numpy as np
import pytest

from backend import nova

REC = {
    "ranked": [{"call": {"coverage": "Cover-3", "pass_rushers": 4, "defenders_in_box": 6},
                "pressure_probability": 0.41, "expected_yards": 5.2}],
    "yards_delta_vs_typical": -0.8,
}


def test_property_25_explanation_length_cap():
    """Feature: coach-pass-rush-assistant, Property 25: Explanation length cap"""
    rng = np.random.default_rng(25)
    for _ in range(100):
        n = int(rng.integers(1, 5000))
        text = "".join(rng.choice(list("abc áñ"), size=n)) + "x"  # termina en no-espacio
        out = nova.explain(REC, invoke=lambda *a, t=text: t)
        assert out.available and 1 <= len(out.text) <= 1000


def test_failure_maps_to_unavailable():
    def boom(*a):
        raise RuntimeError("detalle")

    out = nova.explain(REC, invoke=boom)
    assert not out.available and out.text is None and out.message == nova.UNAVAILABLE_MESSAGE


def test_timeout_maps_to_unavailable():
    start = time.monotonic()
    out = nova.explain(REC, invoke=lambda *a: time.sleep(1.0) or "tarde", timeout=0.1)
    assert not out.available and time.monotonic() - start < 0.8


@pytest.mark.parametrize("bad", ["", "   ", None, 5])
def test_empty_or_non_text_response_is_unavailable(bad):
    assert not nova.explain(REC, invoke=lambda *a: bad).available


def test_bad_recommendation_is_unavailable_without_calling_bedrock():
    called = []
    out = nova.explain({"ranked": []}, invoke=lambda *a: called.append(1) or "x")
    assert not out.available and not called


def test_prompt_mentions_call_probability_yards():
    p = nova.build_prompt(REC)
    assert "Cover-3" in p and "41.0%" in p and "5.2" in p


def test_defaults_when_env_missing_or_empty(monkeypatch):
    seen = []
    for value in (None, ""):
        for var in ("AWS_REGION", "BEDROCK_MODEL_ID"):
            if value is None:
                monkeypatch.delenv(var, raising=False)
            else:
                monkeypatch.setenv(var, value)
        nova.explain(REC, invoke=lambda p, r, m: seen.append((r, m)) or "ok")
    assert seen == [(nova.DEFAULT_REGION, nova.DEFAULT_MODEL_ID)] * 2


class RecordingEnv(Mapping):
    def __init__(self, data):
        self.data, self.reads = data, []

    def __getitem__(self, k):
        self.reads.append(k)
        return self.data[k]

    def get(self, k, default=None):
        self.reads.append(k)
        return self.data.get(k, default)

    def __iter__(self):
        return iter(self.data)

    def __len__(self):
        return len(self.data)


def test_client_created_with_region_only_and_no_credentials_read(monkeypatch):
    env = RecordingEnv({"AWS_REGION": "us-west-2", "BEDROCK_MODEL_ID": "modelo-x",
                        "AWS_ACCESS_KEY_ID": "no-leer", "AWS_SECRET_ACCESS_KEY": "no-leer"})
    monkeypatch.setattr(nova, "os", types.SimpleNamespace(environ=env))
    calls = {}

    class FakeClient:
        def converse(self, **kw):
            calls["converse"] = kw
            return {"output": {"message": {"content": [{"text": "Explicación"}]}}}

    def fake_client(*args, **kwargs):
        calls["client"] = (args, kwargs)
        return FakeClient()

    monkeypatch.setitem(sys.modules, "boto3", types.SimpleNamespace(client=fake_client))
    out = nova.explain(REC)
    assert out.available and out.text == "Explicación"
    assert calls["client"] == (("bedrock-runtime",), {"region_name": "us-west-2"})
    assert calls["converse"]["modelId"] == "modelo-x"
    assert set(env.reads) <= {"AWS_REGION", "BEDROCK_MODEL_ID"}
