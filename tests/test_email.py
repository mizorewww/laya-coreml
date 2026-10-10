"""Email cleaning: a disclaimer footer must not delete the sender's actual request.

Regression tests ported from upstream v0.3.5 (#94). `_DISCLAIMER` used to be applied to
whole paragraphs, so any paragraph that merely *mentioned* boilerplate was deleted outright.
"""

import pytest

from laya_coreml.email import clean_email_body, email_state

DISCLAIMER = "This email is confidential and intended solely for the named addressee."


@pytest.mark.parametrize(
    "body,want",
    [
        # the request survives the footer
        (
            "My account is locked.\n%s\nPlease unlock it." % DISCLAIMER,
            "My account is locked. Please unlock it.",
        ),
        (
            "My account is locked\n%s\nPlease unlock it." % DISCLAIMER,
            "My account is locked Please unlock it.",
        ),
        ("My account is locked. %s" % DISCLAIMER, "My account is locked."),
        # a pure footer is still removed
        ("My account is locked.\n\n%s" % DISCLAIMER, "My account is locked."),
        (
            "My account is locked.\n\nThis email and any files transmitted with it are\n"
            "confidential and intended solely for the named addressee.",
            "My account is locked.",
        ),
        (
            "Please reopen ticket 4411.\n\nIf you have received this message in error, delete it.",
            "Please reopen ticket 4411.",
        ),
        # unrelated cleaning is unchanged
        (
            "Thanks for the update.\nOn Mon, Sep 20, Bob wrote:\n> original text",
            "Thanks for the update.",
        ),
        (
            "Hi team,\nCan you confirm the refund?\nRegards,\nAlice",
            "Hi team,\nCan you confirm the refund?",
        ),
        ("", ""),
    ],
)
def test_clean_email_body(body, want):
    assert clean_email_body(body) == want


def test_email_state_body_keeps_the_request():
    state = email_state("Locked out", "My account is locked. %s" % DISCLAIMER)
    assert state["body"] == "My account is locked."
