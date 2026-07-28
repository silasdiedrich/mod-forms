"""Import and browse modular forms from the LMFDB (https://www.lmfdb.org).

This uses the LMFDB's generic REST API, which exposes each backing
database collection at

    https://www.lmfdb.org/api/<collection>/?field=value&...&_format=json

Two collections are used here:

- ``mf_newforms``: one document per Galois orbit of newforms. Used for
  *browsing/searching* (label, level, weight, dimension, ...).
- ``mf_hecke_cc``: complex numerical embeddings of Hecke eigenvalues, one
  document per embedded newform (labels like "5.4.a.a.1.1", exactly the
  kind of label the paper links to). This is the same numerical data the
  LMFDB itself uses to draw its own plots, and is what lets us evaluate
  f(z) = sum a_n q^n as an actual complex-valued q-expansion.

Caveat: this was written against the LMFDB API's documented shape, but
could not be exercised against a live server in the environment it was
written in (that sandbox's network policy blocks outbound requests to
lmfdb.org). The field names below are isolated as module-level constants
so a mismatch is a one-line fix; if a live response doesn't have a field
we expect, the functions here raise ``LMFDBSchemaError`` with the *actual*
keys found, rather than failing silently or with a bare KeyError.
"""

import json
import urllib.error
import urllib.parse
import urllib.request

from .qexpansion import QExpansion

API_BASE = "https://www.lmfdb.org/api"

NEWFORMS_COLLECTION = "mf_newforms"
HECKE_CC_COLLECTION = "mf_hecke_cc"

# Field names within mf_hecke_cc documents. Isolated here in case the
# live schema differs from what's assumed.
CC_LABEL_FIELD = "label"
CC_AN_FIELD = "an"  # list of {"re": ..., "im": ...} for n = 2, 3, ...
CC_A1_IS_ONE = True  # Hecke-normalized newforms always have a_1 = 1


class LMFDBError(RuntimeError):
    """Network/HTTP problem talking to the LMFDB."""


class LMFDBSchemaError(LMFDBError):
    """The response didn't have the field(s) we expected."""


def _get(collection, params, timeout=30):
    query = urllib.parse.urlencode({**params, "_format": "json"})
    url = f"{API_BASE}/{collection}/?{query}"
    request = urllib.request.Request(url, headers={"User-Agent": "modforms/0.1 (+https://arxiv.org/abs/2002.05234)"})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            payload = response.read().decode("utf-8")
    except urllib.error.URLError as exc:
        raise LMFDBError(f"could not reach the LMFDB at {url}: {exc}") from exc
    try:
        return json.loads(payload), url
    except json.JSONDecodeError as exc:
        raise LMFDBError(f"non-JSON response from {url}: {exc}") from exc


def _results(data, url):
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        for key in ("data", "results"):
            if key in data:
                return data[key]
    raise LMFDBSchemaError(
        f"unrecognized API response shape from {url}: "
        f"{list(data)[:10] if isinstance(data, dict) else type(data)}"
    )


def search_newforms(level=None, weight=None, limit=20, **extra_filters):
    """Browse/search newforms in the LMFDB.

    Returns a list of dicts (raw API documents from ``mf_newforms``),
    each with at least a ``label``. Any extra keyword arguments are passed
    through as additional query filters, e.g. ``char_orbit_label="a"``.

    Example
    -------
    >>> search_newforms(level=105, weight=2)  # doctest: +SKIP
    """
    params = {"_limit": limit}
    if level is not None:
        params["level"] = level
    if weight is not None:
        params["weight"] = weight
    params.update(extra_filters)

    data, url = _get(NEWFORMS_COLLECTION, params)
    results = _results(data, url)
    if results and "label" not in results[0]:
        raise LMFDBSchemaError(
            f"expected a 'label' field in {NEWFORMS_COLLECTION} results from {url}, "
            f"got keys: {sorted(results[0])}"
        )
    return results


def print_newforms(level=None, weight=None, limit=20, **extra_filters):
    """Convenience wrapper around :func:`search_newforms` that prints a
    quick table (label, level, weight, dim) instead of returning data.
    """
    results = search_newforms(level=level, weight=weight, limit=limit, **extra_filters)
    if not results:
        print("(no results)")
        return
    for doc in results:
        label = doc.get("label", "?")
        lvl = doc.get("level", "?")
        wt = doc.get("weight", "?")
        dim = doc.get("dim", "?")
        print(f"{label:20} level={lvl:<6} weight={wt:<4} dim={dim}")


def _normalize_embedded_label(label):
    """Accept either a Galois-orbit label ("5.4.a.a") or a fully embedded
    label ("5.4.a.a.1.1"). For orbit labels, assume the first embedding
    (".1.1"), which is the only embedding when the newform has rational
    (dim=1) coefficients.
    """
    parts = label.split(".")
    if len(parts) == 4:
        return f"{label}.1.1"
    if len(parts) == 6:
        return label
    raise ValueError(
        f"expected a newform label like '5.4.a.a' or an embedded label like "
        f"'5.4.a.a.1.1', got {label!r}"
    )


def fetch_qexpansion(label, n_terms=400):
    """Fetch a specific embedding of a newform's q-expansion from the
    LMFDB and return it as a :class:`~modforms.QExpansion`, ready to plot.

    ``label`` may be a Galois-orbit label ("5.4.a.a", the LMFDB label used
    in the paper) or a fully embedded label ("5.4.a.a.1.1").
    """
    embedded_label = _normalize_embedded_label(label)
    data, url = _get(HECKE_CC_COLLECTION, {CC_LABEL_FIELD: embedded_label, "_limit": 1})
    results = _results(data, url)
    if not results:
        raise LMFDBError(f"no data found for {embedded_label!r} in {HECKE_CC_COLLECTION} ({url})")
    doc = results[0]
    if CC_AN_FIELD not in doc:
        raise LMFDBSchemaError(
            f"expected an {CC_AN_FIELD!r} field in the {HECKE_CC_COLLECTION} document "
            f"for {embedded_label!r}, got keys: {sorted(doc)}. "
            f"Update CC_AN_FIELD (and friends) in modforms/lmfdb.py to match."
        )

    an = doc[CC_AN_FIELD]
    coeffs = [1.0 + 0.0j] if CC_A1_IS_ONE else []
    remaining = n_terms - len(coeffs)
    for entry in an[:remaining]:
        coeffs.append(complex(entry["re"], entry["im"]))

    return QExpansion(coeffs, start=1, label=embedded_label)
