"""Routing, language-detection and temperature-clamp tests ported from upstream v0.3.5.

No model weights are loaded: `Router.route` is pure.
"""

import pytest

from laya_coreml import Router
from laya_coreml.common import QTYPES, TEMP_MAX, TEMP_MIN, clamp_temperature, temp_bucket
from laya_coreml.lang import analyse, detect_script, guess_latin_language, is_english, state_text
from laya_coreml.router import normalise_name

# --------------------------------------------------------------------- script detection


@pytest.mark.parametrize(
    "text,want",
    [
        ("The customer was charged twice and wants a refund.", "latin"),
        ("Հայերեն", "armenian"),
        ("ՀԱՅԵՐԵՆ", "armenian"),
        ("։֊", "unknown"),
        ("Le client a été facturé deux fois et demande un remboursement.", "latin"),
        ("ग्राहक से दो बार शुल्क लिया गया और वह धनवापसी चाहता है।", "devanagari"),
        ("お客様は二重に請求されたため返金を希望しています。", "kana"),
        ("客户被重复扣款要求退款", "han"),
        ("고객이 두 번 청구되어 환불을 원합니다", "hangul"),
        ("تم خصم المبلغ مرتين من العميل ويريد استرداد الأموال", "arabic"),
        ("С клиента дважды сняли деньги и он хочет возврат", "cyrillic"),
        ("Ο πελάτης χρεώθηκε δύο φορές και θέλει επιστροφή χρημάτων", "greek"),
        ("הלקוח חויב פעמיים ורוצה החזר כספי", "hebrew"),
        ("", "unknown"),
        ("12345 6789", "unknown"),
    ],
)
def test_detect_script(text, want):
    assert detect_script(text) == want


# --------------------------------------------------------------------- english vs not


@pytest.mark.parametrize(
    "text,want",
    [
        ("Please refund the duplicate charge on invoice 4411 today.", True),
        ("Հայերեն", False),
        ("refund me", True),
        ("ग्राहक से दो बार शुल्क लिया गया", False),
        ("お客様は二重に請求されました", False),
        (
            "Le client a été facturé deux fois et il demande un remboursement pour la "
            "facture qui a été payée le mois dernier avec la carte de crédit",
            False,
        ),
        (
            "Der Kunde wurde zweimal belastet und möchte eine Rückerstattung für die "
            "Rechnung die nicht korrekt ist und auch nicht bezahlt wurde",
            False,
        ),
        # Latin-script languages with no stopword list of their own: reported upstream in #35,
        # where Romanian states were handed to the English checkpoint instead of the
        # multilingual one. An unidentified language must never be assumed English.
        ("Gătește-mi o rețetă de sarmale de post pentru mâine.", False),
        ("Am fost taxat de două ori pentru factura din luna martie și vreau banii", False),
        ("Klient został obciążony dwukrotnie i chce zwrot pieniędzy za fakturę", False),
        ("Zákazníkovi byla částka účtována dvakrát a žádá o vrácení peněz", False),
        ("Müşteriden iki kez ücret alındı ve para iadesi istiyor lütfen yardım", False),
        ("Khách hàng đã bị thu phí hai lần và muốn được hoàn tiền ngay", False),
        # English with the odd loanword must not tip over into the multilingual checkpoint
        (
            "We visited a cafe in Zurich and the naive assumption about the "
            "invoice was wrong, so please refund the duplicate charge",
            True,
        ),
    ],
)
def test_is_english(text, want):
    assert is_english(text) is want


def test_undecided_latin_is_not_dressed_up_as_a_detection():
    # A single shared function word used to name a language ("para" in Turkish text was
    # called Spanish). Undecided must be reported as undecided.
    det = analyse("Müşteriden iki kez ücret alındı ve para iadesi istiyor")
    assert det["language_undecided"] is True
    assert det["language"] is None
    assert (
        analyse("Please refund the duplicate charge on the invoice")["language_undecided"] is False
    )
    assert analyse("Gătește-mi o rețetă de sarmale")["diacritic_rate"] > 0.02
    assert analyse("Please refund the duplicate charge today")["diacritic_rate"] == 0.0


def test_analyse_reports_the_same_keys_from_every_branch():
    keys = {
        "script",
        "script_profile",
        "language",
        "is_english",
        "language_undecided",
        "diacritic_rate",
        "non_latin_fraction",
        "mixed_segment",
    }
    for text in (
        "Please refund the duplicate charge",
        "ग्राहक से दो बार",
        "Gătește-mi o rețetă de sarmale",
        "12345 ???",
    ):
        assert set(analyse(text)) == keys


def test_zero_stopword_tie_invents_no_language():
    assert guess_latin_language("Cât e ora acum la Tokyo") is None


def test_known_gap_romanian_without_diacritics_still_reads_english():
    # Kept visible on purpose, as upstream does: short Romanian with no diacritics and an
    # English function word ("in") still reads as English. A real LID model is the fix.
    assert is_english("Care este ora in Tokyo?") is True


@pytest.mark.parametrize(
    "text,want",
    [
        ("The customer was charged twice and wants a refund for this invoice", "en"),
        ("Le client a ete facture deux fois et il demande un remboursement pour la facture", "fr"),
        (
            "Der Kunde wurde zweimal belastet und moechte eine Rueckerstattung fuer die Rechnung",
            "de",
        ),
        (
            "El cliente fue cobrado dos veces y quiere que le devuelvan el dinero por la factura",
            "es",
        ),
        ("refund", None),
    ],
)
def test_guess_latin_language(text, want):
    assert guess_latin_language(text) == want


def test_state_text_flattens_and_ignores_keys():
    assert "charged twice" in state_text({"body": "charged twice", "n": 3})
    assert "deep" in state_text({"a": {"b": ["deep"]}})
    assert "x" in state_text(["x", {"y": "z"}])
    assert state_text(None) == ""
    det = analyse({"subject": "नमस्ते", "body": "ग्राहक से दो बार शुल्क लिया गया"})
    assert det["is_english"] is False


def test_script_profile_armenian():
    assert analyse("Հայերեն")["script_profile"] == {"armenian": 1.0}
    assert analyse("Հայերեն abc")["non_latin_fraction"] == 0.7


# --------------------------------------------------------------------- routing decisions

Q_GENERIC = {
    "dept": {
        "type": "choice",
        "instructions": "Which team?",
        "criteria": {"billing": None, "tech": None},
    }
}


@pytest.mark.parametrize(
    "text",
    [
        "Gătește-mi o rețetă de sarmale de post pentru mâine.",
        "Exportă APK-ul pentru Android și pune-l pe Drive ca să-l instalez.",
        "Klient został obciążony dwukrotnie i chce zwrot pieniędzy za fakturę",
        "Müşteriden iki kez ücret alındı ve para iadesi istiyor lütfen yardım",
    ],
)
def test_unidentified_latin_routes_to_multilingual(text):
    assert Router().route(text).model == "multilingual"


def test_unidentified_latin_reason_says_what_it_routed_on():
    reason = Router().route("Müşteriden iki kez ücret alındı ve para iadesi istiyor").reason
    assert "not identified" in reason


def test_identified_non_english_reason_names_the_language():
    reason = (
        Router()
        .route(
            "Der Kunde wurde zweimal belastet und moechte eine Rueckerstattung fuer die "
            "Rechnung die nicht korrekt ist"
        )
        .reason
    )
    assert "'de'" in reason


def test_english_routing_unchanged():
    router = Router()
    assert (
        router.route("Please refund the duplicate charge on invoice 4411 today.").model == "english"
    )
    assert router.route("refund me").model == "multilingual"  # short text is undecided
    assert Router(default="english").route("refund me").model == "english"
    assert router.route("Հայերեն", model="english").model == "english"
    assert normalise_name("ML") == "multilingual"


# --------------------------------------------------------------------- temperature clamp (#35)


@pytest.mark.parametrize(
    "value,want",
    [
        (0.1006, 0.5),  # pathological sharpening
        (0.10058280825614929, TEMP_MIN),  # the shipped choice:11+ bucket
        (1.7601518630981445, 1.7601518630981445),  # legitimate value untouched
        (1.0, 1.0),
        (9.0, TEMP_MAX),
        (0.0, TEMP_MIN),
        (-3.0, TEMP_MIN),
        (None, 1.0),
        ("x", 1.0),
        (float("nan"), 1.0),
        (float("inf"), 1.0),
    ],
)
def test_clamp_temperature(value, want):
    assert clamp_temperature(value) == want


def test_clamp_bounds_are_sane_and_bucket_matches_reported_case():
    assert TEMP_MIN <= 1.0 <= TEMP_MAX
    # 13 options is the bucket the reported skill-router landed in
    assert temp_bucket(QTYPES["choice"], 13) == "choice:11+"
