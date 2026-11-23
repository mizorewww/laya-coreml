"""v0.5.3 boundary regressions that run without native Core ML or model weights.

Upstream 0.5.3 adds unaccented and closed-class Spanish and Portuguese stopwords, so short
ASCII-normalised messages are detected and routed to the multilingual model, and adds the elided
Italian `qual`. Upstream's own assertions run unchanged through the `router` suite wrapper; these
pin the ported tables to the reference checkout and a few cases without it.

The release also translates vegaml's string-keyed choice labels back to the caller's labels in
`laya.vega`, which this distribution does not ship.
"""

import importlib.util
from pathlib import Path

import pytest

from laya_coreml import Router, lang


@pytest.mark.parametrize(
    ("language", "text"),
    [
        ("es", "no puedo entrar a mi cuenta"),
        ("es", "estoy esperando desde ayer una respuesta"),
        ("pt", "como faco para reembolsar?"),
        ("pt", "qual o prazo de entrega amanha?"),
    ],
)
def test_short_unaccented_spanish_and_portuguese_route_multilingual(language, text):
    assert lang.guess_latin_language(text) == language
    assert lang.is_english(text) is False
    assert Router(default="english").route(text).model == "multilingual"
    mixed = {"header": "Customer support ticket received from web form", "body": text}
    assert Router().route(mixed).model == "multilingual"


def test_italian_qual_stays_italian():
    assert lang.guess_latin_language("qual è l'ultima notizia da rete quattro") == "it"


@pytest.mark.parametrize(
    "text",
    [
        "Please send me the MIS report before tomorrow morning",
        "I ate lunch and did not spend a dime on the trip",
        "Do me a favor and check the solo plan settings",
    ],
)
def test_english_controls_stay_english(text):
    assert lang.is_english(text) is True
    assert Router().route(text).model == "english"


def test_stopword_tables_match_upstream():
    path = Path(__file__).resolve().parents[1] / ".upstream" / "laya" / "lang.py"
    if not path.exists():
        pytest.skip("checkout pinned upstream into .upstream (CI does this)")
    spec = importlib.util.spec_from_file_location("_official_lang", path)
    official = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(official)
    assert lang._STOP == official._STOP
