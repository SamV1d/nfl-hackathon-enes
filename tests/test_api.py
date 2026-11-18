"""Pruebas de la API (7.4). Datos sintéticos escritos en un directorio temporal."""
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

from backend import config, nova
from backend.loader import REQUIRED_FILES
from backend.main import create_app

from .fixtures import write_csv_dir

GOOD = {"down": 3, "yards_to_go": 5, "offense_formation": "SHOTGUN", "expected_coverage": "Cover-3"}


@pytest.fixture(scope="module")
def data_dir(tmp_path_factory):
    d = tmp_path_factory.mktemp("data")
    write_csv_dir(d, np.random.default_rng(7), n_plays=500)
    return d


@pytest.fixture(scope="module")
def client(data_dir, tmp_path_factory):
    static = tmp_path_factory.mktemp("dist")
    (static / "index.html").write_text("<html>dashboard</html>")
    return TestClient(create_app(data_dir, static))


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200 and r.json() == {"status": "ok"}


def test_routes_only_api_health_and_static_root(client):
    paths = [getattr(r, "path", "") for r in client.app.routes]
    outside = [p for p in paths if not (p.startswith("/api") or p == "/health" or p == "")]
    assert outside == []
    assert "" in paths  # montaje estático en la raíz, registrado al final
    assert paths[-1] == ""


def test_root_serves_dashboard_and_api_on_same_app(client):
    assert "dashboard" in client.get("/").text
    assert client.get("/api/teams").status_code == 200


def test_recommend_ok_and_ranked(client):
    vocab = client.get("/api/vocab").json()
    body = {**GOOD, "offense_formation": vocab["offense_formations"][0],
            "expected_coverage": vocab["expected_coverages"][0]}
    r = client.post("/api/recommend", json=body)
    assert r.status_code == 200
    j = r.json()
    assert j["best_call"] == j["ranked"][0] and j["recommendation_id"]
    assert all(0 <= x["pressure_probability"] <= 1 for x in j["ranked"])
    assert isinstance(j["low_confidence"], bool) and j["two_point_conversion"] is False


def test_recommend_two_point_flag(client):
    r = client.post("/api/recommend", json={**GOOD, "down": 0})
    assert r.status_code == 200 and r.json()["two_point_conversion"] is True


@pytest.mark.parametrize("patch,fields", [
    ({"down": 5}, ["down"]), ({"down": -1}, ["down"]), ({"yards_to_go": 100}, ["yards_to_go"]),
    ({"offense_formation": "NOPE"}, ["offense_formation"]), ({"expected_coverage": "NOPE"}, ["expected_coverage"]),
    ({"down": 9, "expected_coverage": ""}, ["down", "expected_coverage"]),
    ({"down": "x"}, ["down"]),
])
def test_recommend_422_names_invalid_fields(client, patch, fields):
    r = client.post("/api/recommend", json={**GOOD, **patch})
    assert r.status_code == 422 and r.json()["fields"] == fields


def test_recommend_422_missing_fields(client):
    r = client.post("/api/recommend", json={})
    assert r.status_code == 422
    assert r.json()["fields"] == ["down", "expected_coverage", "offense_formation", "yards_to_go"]


def test_metrics(client):
    j = client.get("/api/metrics").json()
    assert j["pressure_model"]["n_train"] + j["pressure_model"]["n_heldout"] == j["yards_model"]["n_train"] + j["yards_model"]["n_heldout"]
    assert "mae" in j["pressure_model"] and "mae_yards" in j["yards_model"]


def test_games_sorted_and_replay(client):
    games = client.get("/api/games").json()
    assert [g["date"] for g in games] == sorted(g["date"] for g in games) and len(games) == 8
    with_plays = client.get(f"/api/games/{games[0]['game_id']}/plays")
    assert with_plays.status_code == 200
    empty = [g for g in games if client.get(f"/api/games/{g['game_id']}/plays").json()["total"] == 0]
    assert len(empty) == 1
    e = client.get(f"/api/games/{empty[0]['game_id']}/plays").json()
    assert e["plays"] == [] and e["message"]
    assert client.get("/api/games/1/plays").status_code == 404


def test_team_report_and_empty_team(client):
    teams = client.get("/api/teams").json()["teams"]
    assert teams
    rep = client.get(f"/api/teams/{teams[0]}/report").json()
    assert rep["has_data"] and rep["league_average"]["pass_plays"] > 0
    assert len(rep["top_sackers"]) <= 5
    empty = client.get("/api/teams/ZZZ/report")
    assert empty.status_code == 200 and empty.json()["has_data"] is False and empty.json()["message"]
    unknown_ids = [p for rep_ in (rep["top_sackers"] + rep["top_pressure"]) for p in [rep_] if p["position"] == "unknown"]
    assert all(p["name"] == str(p["player_id"]) for p in unknown_ids)


def test_findings(client):
    f = client.get("/api/findings").json()
    assert set(f) == {"summary", "by_coverage", "by_formation", "by_defenders_in_box", "top_sackers", "top_teams"}
    assert 0 < f["summary"]["pressure_rate"] < 1


def test_explain_unknown_id_is_404_and_never_calls_nova(client, monkeypatch):
    monkeypatch.setattr(nova, "explain", lambda *a, **k: pytest.fail("no debe invocar Nova"))
    r = client.post("/api/recommendations/does-not-exist/explain")
    assert r.status_code == 404


def test_explain_ok_and_failure(client, monkeypatch):
    rid = client.post("/api/recommend", json=GOOD).json()["recommendation_id"]
    before = client.post("/api/recommend", json=GOOD).json()  # otra recomendación; no debe afectar
    monkeypatch.setattr(nova, "explain", lambda rec, **k: nova.Explanation(True, text="x" * 10))
    ok = client.post(f"/api/recommendations/{rid}/explain")
    assert ok.status_code == 200 and ok.json() == {"explanation": "x" * 10}
    monkeypatch.setattr(nova, "explain", lambda rec, **k: nova.Explanation(False, message=nova.UNAVAILABLE_MESSAGE))
    bad = client.post(f"/api/recommendations/{rid}/explain")
    assert bad.status_code == 503 and bad.json()["detail"] == nova.UNAVAILABLE_MESSAGE
    assert client.get("/api/teams").status_code == 200  # el resto sigue funcionando
    assert before["recommendation_id"] != rid


def test_recommend_never_touches_nova(client, monkeypatch):
    monkeypatch.setattr(nova, "explain", lambda *a, **k: pytest.fail("no debe invocar Nova"))
    assert client.post("/api/recommend", json=GOOD).status_code == 200


def test_model_unavailable_gives_503_but_other_endpoints_work(tmp_path):
    write_csv_dir(tmp_path, np.random.default_rng(3), n_plays=60)  # menos de 100 jugadas etiquetadas
    c = TestClient(create_app(tmp_path, tmp_path / "no-dist"))
    for method, url in (("post", "/api/recommend"), ("get", "/api/metrics"), ("get", "/api/vocab")):
        r = getattr(c, method)(url, **({"json": GOOD} if method == "post" else {}))
        assert r.status_code == 503 and r.json()["detail"] == "model unavailable"
    assert c.get("/health").json() == {"status": "ok"}
    assert c.get("/api/teams").status_code == 200
    assert c.get("/api/findings").status_code == 200
    games = c.get("/api/games").json()
    plays = c.get(f"/api/games/{games[0]['game_id']}/plays").json()["plays"]
    assert all(p["best_call"] is None for p in plays)


@pytest.mark.parametrize("present", [[], ["games.csv"], ["plays.csv", "players.csv"]])
def test_fallback_lists_all_missing_files(tmp_path, present):
    for f in present:
        (tmp_path / f).write_text("gameId,playId,nflId,pff_role\n1,1,1,x\n")
    app = create_app(tmp_path, tmp_path / "no-dist")
    c = TestClient(app)
    missing = [f for f in REQUIRED_FILES if f not in present]
    for method, url in (("get", "/api/teams"), ("post", "/api/recommend"), ("get", "/api/anything/else")):
        r = getattr(c, method)(url)
        assert r.status_code == 503
        assert r.json()["missing_files"] == missing
        assert all(name in r.json()["detail"] for name in missing)
    assert c.get("/health").json() == {"status": "ok"}
    assert app.state.startup_error is not None


def test_unreadable_file_reports_name_and_category(tmp_path):
    for f in REQUIRED_FILES:
        (tmp_path / f).write_text("gameId,playId,nflId,pff_role\n1,1,1,x\n")
    (tmp_path / "games.csv").write_text("")
    r = TestClient(create_app(tmp_path, tmp_path / "x")).get("/api/teams")
    assert r.status_code == 503 and (r.json()["file"], r.json()["category"]) == ("games.csv", "empty")


REAL = config.PROJECT_ROOT / "Datos" / "data"


@pytest.mark.skipif(not all((REAL / f).is_file() for f in REQUIRED_FILES), reason="datos reales ausentes")
def test_real_data_reference_values():
    c = TestClient(create_app(REAL, Path("no-dist")))
    f = c.get("/api/findings").json()
    s = f["summary"]
    assert s["pressure_rate"] == pytest.approx(0.371, abs=0.01)  # EDA: 37,6%
    assert s["avg_yards_with_pressure"] == pytest.approx(3.94, abs=0.1)
    assert s["avg_yards_without_pressure"] == pytest.approx(8.15, abs=0.1)
    assert s["sack_rate_with_pressure"] == pytest.approx(0.17, abs=0.01)
    assert s["sack_rate_without_pressure"] == 0
    assert [(p["name"], p["count"]) for p in f["top_sackers"][:3]] == [
        ("Myles Garrett", 12), ("Trey Hendrickson", 10), ("Matt Judon", 9)]
    assert len(c.get("/api/games").json()) == 122

def test_model_audit_endpoint_degrades_gracefully_on_synthetic_data(client):
    r = client.get("/api/model-audit")
    assert r.status_code in (200, 503)
    if r.status_code == 503:
        assert r.json()["detail"] == "model audit unavailable"
    assert client.get("/api/model-audit").status_code == r.status_code  # el fallo no se reintenta en bucle
