"""Tests for the LMFDB parsing logic, against mocked API responses.

The mf_hecke_cc fixture below mirrors an actual live response captured for
"105.2.a.a.1.1" (a weight-2 newform): wrapped as {"data": [...]}, with
an_normalized as a list of [re, im] pairs starting at n=1. LMFDBSchemaError
is what should fire if some other query returns a differently-shaped
document; that behavior is also tested here.
"""

import pytest

from modforms import lmfdb


def test_search_newforms_parses_results(monkeypatch):
    fixture = [
        {"label": "105.2.a.a", "level": 105, "weight": 2, "dim": 1},
        {"label": "105.2.a.b", "level": 105, "weight": 2, "dim": 2},
    ]

    def fake_get(collection, params, timeout=30):
        assert collection == lmfdb.NEWFORMS_COLLECTION
        assert params["level"] == 105
        assert params["weight"] == 2
        return fixture, "http://fake/url"

    monkeypatch.setattr(lmfdb, "_get", fake_get)
    results = lmfdb.search_newforms(level=105, weight=2)
    assert results == fixture


def test_search_newforms_schema_error(monkeypatch):
    def fake_get(collection, params, timeout=30):
        return [{"unexpected_field": 1}], "http://fake/url"

    monkeypatch.setattr(lmfdb, "_get", fake_get)
    with pytest.raises(lmfdb.LMFDBSchemaError):
        lmfdb.search_newforms(level=1)


def test_fetch_qexpansion_unnormalizes_real_schema(monkeypatch):
    # An actual live response for 105.2.a.a.1.1 (weight 2), truncated to 6
    # terms. an_normalized[n-1] = a_n / n**((weight-1)/2).
    fixture = {
        "data": [
            {
                "label": "105.2.a.a.1.1",
                "weight": 2,
                "level": 105,
                "an_normalized": [
                    [1.0, 0.0],
                    [0.7071067811865476, 0.0],
                    [0.5773502691896257, 0.0],
                    [-0.5, 0.0],
                    [0.4472135954999579, 0.0],
                    [0.408248290463863, 0.0],
                ],
            }
        ]
    }

    def fake_get(collection, params, timeout=30):
        assert collection == lmfdb.HECKE_CC_COLLECTION
        assert params[lmfdb.CC_LABEL_FIELD] == "105.2.a.a.1.1"
        return fixture, "http://fake/url"

    monkeypatch.setattr(lmfdb, "_get", fake_get)
    form = lmfdb.fetch_qexpansion("105.2.a.a", n_terms=6)
    assert form.label == "105.2.a.a.1.1"
    assert form.weight == 2
    assert form.level == 105
    # Un-normalized, these should come out as clean integers: 1,1,1,-1,1,1.
    for coeff, expected in zip(form.coeffs, [1, 1, 1, -1, 1, 1]):
        assert abs(coeff - expected) < 1e-9


def test_fetch_qexpansion_accepts_dict_shaped_an_entries(monkeypatch):
    fixture = [
        {
            "label": "5.4.a.a.1.1",
            "weight": 4,
            "an_normalized": [
                {"re": 1.0, "im": 0.0},
                {"re": -0.5, "im": 1.0},
            ],
        }
    ]

    def fake_get(collection, params, timeout=30):
        return fixture, "http://fake/url"

    monkeypatch.setattr(lmfdb, "_get", fake_get)
    form = lmfdb.fetch_qexpansion("5.4.a.a", n_terms=2)
    # weight=4 => n**((4-1)/2) = n**1.5; n=2: (-0.5+1.0j) * 2**1.5
    assert abs(form.coeffs[0] - 1.0) < 1e-9
    assert abs(form.coeffs[1] - (-0.5 + 1.0j) * 2**1.5) < 1e-9


def test_fetch_qexpansion_missing_an_field_raises_schema_error(monkeypatch):
    def fake_get(collection, params, timeout=30):
        return [{"label": "5.4.a.a.1.1", "weight": 4, "some_other_field": 1}], "http://fake/url"

    monkeypatch.setattr(lmfdb, "_get", fake_get)
    with pytest.raises(lmfdb.LMFDBSchemaError):
        lmfdb.fetch_qexpansion("5.4.a.a")


def test_fetch_qexpansion_missing_weight_field_raises_schema_error(monkeypatch):
    def fake_get(collection, params, timeout=30):
        return [{"label": "5.4.a.a.1.1", "an_normalized": [[1.0, 0.0]]}], "http://fake/url"

    monkeypatch.setattr(lmfdb, "_get", fake_get)
    with pytest.raises(lmfdb.LMFDBSchemaError):
        lmfdb.fetch_qexpansion("5.4.a.a")


def test_normalize_embedded_label():
    assert lmfdb._normalize_embedded_label("5.4.a.a") == "5.4.a.a.1.1"
    assert lmfdb._normalize_embedded_label("5.4.a.a.1.1") == "5.4.a.a.1.1"
    with pytest.raises(ValueError):
        lmfdb._normalize_embedded_label("not-a-label")
