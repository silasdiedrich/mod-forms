"""Tests for the LMFDB parsing logic, against mocked API responses.

These verify that our code correctly turns an API response *shaped like
what we assume the LMFDB returns* into a QExpansion/search results. They
cannot verify that assumption is correct against a live server (this
environment has no network access to lmfdb.org) -- see the caveat at the
top of modforms/lmfdb.py. LMFDBSchemaError is what should fire if the real
schema differs; that behavior is also tested here.
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


def test_fetch_qexpansion_parses_complex_coefficients(monkeypatch):
    fixture = [
        {
            "label": "5.4.a.a.1.1",
            "an": [
                {"re": 1.5, "im": -2.0},
                {"re": 0.0, "im": 3.25},
            ],
        }
    ]

    def fake_get(collection, params, timeout=30):
        assert collection == lmfdb.HECKE_CC_COLLECTION
        assert params[lmfdb.CC_LABEL_FIELD] == "5.4.a.a.1.1"
        return fixture, "http://fake/url"

    monkeypatch.setattr(lmfdb, "_get", fake_get)
    form = lmfdb.fetch_qexpansion("5.4.a.a", n_terms=10)
    assert form.label == "5.4.a.a.1.1"
    assert form.coeffs[0] == 1 + 0j  # a_1 = 1 by Hecke normalization
    assert form.coeffs[1] == 1.5 - 2.0j
    assert form.coeffs[2] == 0.0 + 3.25j


def test_fetch_qexpansion_missing_an_field_raises_schema_error(monkeypatch):
    def fake_get(collection, params, timeout=30):
        return [{"label": "5.4.a.a.1.1", "some_other_field": 1}], "http://fake/url"

    monkeypatch.setattr(lmfdb, "_get", fake_get)
    with pytest.raises(lmfdb.LMFDBSchemaError):
        lmfdb.fetch_qexpansion("5.4.a.a")


def test_normalize_embedded_label():
    assert lmfdb._normalize_embedded_label("5.4.a.a") == "5.4.a.a.1.1"
    assert lmfdb._normalize_embedded_label("5.4.a.a.1.1") == "5.4.a.a.1.1"
    with pytest.raises(ValueError):
        lmfdb._normalize_embedded_label("not-a-label")
