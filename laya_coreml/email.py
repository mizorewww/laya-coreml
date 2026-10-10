# Derived from Laya (Apache-2.0); see NOTICE. Modified for laya-coreml.
"""Email utilities for cleaning and structuring email inputs in laya.

The markers below cover English, Portuguese, Spanish and French mail clients. The Router already sends
Portuguese and Spanish states to the multilingual checkpoint, but with English-only markers their
cleaning was a no-op: Gmail's `Em ... escreveu:`, Outlook's `-----Mensagem original-----`, the
`Atenciosamente` sign-off and the confidentiality footer all reached the model, and the quoted
history (often a *different* request) weighed on the answer as much as the new message did.
French mail reads the same way through an English-only cleaner: Gmail's `Le ... a écrit :`,
Outlook's `-----Message d'origine-----`, `Cordialement` and `Envoyé depuis mon iPhone` all survived.
"""

import re
import unicodedata
from typing import Dict, List, Optional

# One definition, in the module that holds the other presets. Re-exported here because
# `from laya.email import email_questions` is a path callers already have.
from .presets import email_questions  # noqa: F401

_QUOTE_HEADERS = [
    re.compile(r"^\s*On .{0,300}wrote:\s*$", re.I),
    # "Em resposta ao que você escreveu:" is body text; a client's attribution always carries a date
    re.compile(r"^\s*Em (?=.*\d).{0,300}escreveu:\s*$", re.I),
    re.compile(r"^\s*El (?=.*\d).{0,300}escribi[óo]:\s*$", re.I),
    # Gmail's French attribution always carries a date too ("Le lun. ... a écrit :"); a bare
    # "Le rapport que vous avez écrit :" is body text, the same rule as above. French typography
    # puts a space before the colon, so the verb allows one (`a écrit :`, never `a écrit:`).
    re.compile(r"^\s*Le (?=.*\d).{0,300}a [eé]crit\s*:\s*$", re.I),
    re.compile(r"^\s*-{2,}\s*(Original|Forwarded) Message\s*-{2,}", re.I),
    re.compile(
        r"^\s*-{2,}\s*(Mensagem (original|encaminhada)|Mensaje (original|reenviado)|Message d'origine)\s*-{2,}",
        re.I,
    ),
    re.compile(r"^\s*_{8,}\s*$"),
    # `From:` opens ordinary prose too ("From: my side the integration works, but please
    # refund..."), and a reply header always carries the sender, so the header is only
    # recognised when an address follows -- the same rule as `De:` below. A bare
    # `From: Name` header is caught by _HEADER_FROM_NAME/_HEADER_NEXT instead, which need the
    # header's own `Sent:`/`Date:` line to tell it apart from a sentence.
    re.compile(r"^\s*From:\s.*[@<]", re.I),
    # `De:` also opens ordinary Portuguese/Spanish lines ("De: 10/09 a 15/09"), so the Outlook
    # header is only recognised when it carries an address. French Outlook writes `De :` with a
    # space, so the marker allows one; the address rule is unchanged
    re.compile(r"^\s*De\s*:\s.*[@<]", re.I),
]
# Gmail wraps a long attribution line, leaving `fulano@x.com> escreveu:` alone on the next line.
# That tail cuts too, and takes the `On/Em/El/Le ...` head it belongs to with it. The French tail
# keeps its spaced colon (`support@x.com> a écrit :`).
_ATTRIBUTION_TAIL = re.compile(
    r"^.{0,120}\S@\S+\s+(wrote|escreveu|escribi[óo]|a [eé]crit)\s*:\s*$", re.I
)
_ATTRIBUTION_HEAD = re.compile(r"^\s*(On|Em|El|Le) (?=.*\d)", re.I)
# Exchange often leaves the address out of Outlook's reply header ("De: Maria Souza"), so a bare `De:`
# only cuts when the header's own `Enviado:` line, or a dated `Data:`/`Fecha:` line, follows it.
# `Para:` is not enough: "De: 10/09 / Para: 15/09" is how a leave request reads.
#
# The same is true of a bare English `From: Maria Souza`, which is why the marker above needs
# this rule: the English client lines are the translations of the two `De:` neighbours. A line
# that only looks like prose still has to be told apart from a header by its neighbours, so the
# English pair is "From: <name>" followed by "Sent:"/"Date:". French Outlook writes the same
# header as `De : Marie Dupont` with `Envoyé :` underneath, so the pair gains that translation too.
# Both allow the French spaced colon (`De :`, `Envoyé :`); the neighbours still do the telling apart.
_HEADER_FROM_NAME = re.compile(r"^\s*(De|From)\s*:\s+\S", re.I)
_HEADER_NEXT = re.compile(
    r"^\s*(Enviad[oa]( em| el)?:\s|Envoy[ée]( le)?\s*:\s|Sent:\s|(Data|Fecha|Date):\s.*\d{4})", re.I
)
# A closing's name starts with a letter that is not lowercase: capitalised in any script
# (`Łukasz`, `Дмитрий`) or caseless (`山田`). `re` cannot say "not a lowercase letter in any
# script" -- a class has to list ranges, and `[^\W\d_a-zß-öø-ÿ]` stops at Latin-1, so
# `Thanks, żaneta` read as a name and the line counted as a sign-off. The tail is matched
# structurally instead, and each token's first letter is judged by category below -- the same
# rule as the TS port's `\p{Lu}\p{Lt}\p{Lo}`. Combining marks ride along with the letter before
# them (`Jose\u0301` is `José`), as `\p{M}` allows in the port. `re` has no `\p{M}` either, and a
# class cannot list those ranges any more than it could list the lowercase ones, so a mark that
# rides on a character is dropped before the tail is matched and the letter it rides on answers
# the case question. A mark with no base -- opening the tail, or following a space -- is left
# where it is, so it still breaks the token as the port's leading `\p{Lu}\p{Lt}\p{Lo}` does.
# `\p{M}` is marks only: ZWJ and ZWNJ are `Cf`, so `Thanks, क्‌ष` is kept here and in the port alike.
_SIGNOFF_HEAD = re.compile(
    r"^\s*(?i:best|kind|warmest|warm|many thanks|thanks|thank you|regards|cheers|sincerely)"
    r"(?i:\s+(?:and|&)\s+regards|\s+(?:regards|wishes|again|in advance|a lot|so much|very much))?"
)
_SIGNOFF_TAIL = re.compile(r"^[\s,;:!.]*(?:[^\W\d_][\w'-]*[\s,.]*){0,3}$")
_SIGNOFF_TOKEN = re.compile(r"[^\W\d_][\w'-]*")


def _drop_marks(text: str) -> str:
    """Remove combining marks, the `Mn`/`Mc`/`Me` categories the port spells `\\p{M}`.

    Only a mark that rides on a preceding character goes; one that opens the string or
    follows a space has no base to ride on and stays, so it still separates tokens.
    """
    kept = []
    for ch in text:
        if unicodedata.category(ch).startswith("M") and kept and not kept[-1].isspace():
            continue
        kept.append(ch)
    return "".join(kept)


def _is_english_signoff(line: str) -> bool:
    """True when a closing word is followed by nothing but punctuation and a short name."""
    m = _SIGNOFF_HEAD.match(line)
    if m is None:
        return False
    tail = _drop_marks(line[m.end() :])
    if _SIGNOFF_TAIL.match(tail) is None:
        return False
    return all(
        unicodedata.category(token[0]) in ("Lu", "Lt", "Lo")
        for token in _SIGNOFF_TOKEN.findall(tail)
    )


def _marker_matches(marker, line: str) -> bool:
    """One `_SIGNATURE_MARKERS` entry: a compiled pattern, or a callable for a rule a pattern
    cannot express (the English sign-off)."""
    return bool(marker(line)) if callable(marker) else bool(marker.match(line))


_SIGNATURE_MARKERS = [
    re.compile(r"^\s*--\s*$"),
    # A closing line is the closing word plus punctuation and at most a name. Anything else on
    # the line is a sentence, and the case of the next word is what separates the two: a name is
    # capitalised, "for" in "Thanks for the quick reply." is not. The closing words are matched
    # case-insensitively, the name is not, so the flag is scoped instead of global.
    # `warmest` and `and/& regards` are closings the alternation did not reach; the name that may
    # follow is judged in `_is_english_signoff` above, a callable because `re` cannot express its
    # rule. `Regards, Łukasz` is a sign-off, `Thanks for the reply` is not.
    _is_english_signoff,
    # Portuguese/Spanish sign-offs match only on their own: "Obrigado pelo retorno, mas ..." is a
    # request, not a signature, so unlike the English marker no trailing words are allowed.
    # French closings keep the same rule: "Merci pour votre aide, mais ..." stays, while a bare
    # "Cordialement," or "Merci," goes with the name underneath it.
    re.compile(
        r"^\s*(atenciosamente|att|abraços?|abs|um abraço|cordialmente|grat[oa]|(muito )?obrigad[oa]s?"
        r"( desde já| pela atenção)?|(com os melhores )?cumprimentos|saudações|"
        r"(un )?saludos?( cordiales)?|atentamente|(muchas )?gracias( de antemano)?|"
        r"(bien )?cordialement|salutations( distinguées)?|bien à vous|merci( d'avance)?|"
        r"bonne journée)[\s,!.]*$",
        re.I,
    ),
]
# Mobile and mail-app footers. Only a line that is nothing *but* the footer matches -- "Enviado do meu
# celular o comprovante ontem." is a request -- and such a line may run to 60 characters, since
# Samsung's default ("Enviado do meu smartphone Samsung Galaxy.") is longer than a sign-off's 40.
# Apple's French default ("Envoyé depuis mon iPhone") reads the same way.
_DEVICE = (
    r"iphone|ipad|android|ios|mobile|celular|telemóvel|móvil|galaxy|smartphone|samsung|tablet|"
    r"outlook|yahoo|mail|e-?mail|gmail|windows"
)
_DEVICE_FOOTER = re.compile(
    r"^\s*((enviad[oa] (do|pelo|pela|via|desde|a partir do)( meu| minha| mi)?|sent from( my)?|"
    r"envoy[ée] (depuis|de)( mon| ma| mes)?)"
    r" (%s)( (%s|para|for|no|na|\d+|phone|device|pro|max|mini|plus|using [a-z][a-z0-9_.+-]*))*"
    r"|(obter o|get) outlook (para|for) (ios|android))[\s.!]*$" % (_DEVICE, _DEVICE),
    re.I,
)
_DISCLAIMER = re.compile(
    # English: tied to a disclaimer noun and a disclaimer tail, the way the Portuguese
    # branches below are. The bare word matched any sentence that merely mentioned it,
    # so "Is this confidential?" and "Confidential: I need a refund." were deleted whole.
    # `[^.]` rather than `[^.\n]`: a footer wraps, so "are\nconfidential" must still match.
    r"(\b(e-?mail|message|information|communication|transmission|contents?)\b[^.]{0,60}"
    r"\bconfidential\b[^.]{0,60}\b(intended|solely|addressee|recipient|privileged|"
    r"disclos|unauthori[sz]ed)|"
    r"\bconfidential\b[^.]{0,60}\b(and (may|is) (also )?privileged)|"
    r"if you (have )?received this (e-?mail|message) in error|"
    # Portuguese/Spanish: tied to "this message/e-mail" rather than the bare word `confidencial`,
    # which a sender's own request ("preciso do contrato confidencial") uses just as often
    r"\b(esta|este) (mensagem|e-?mail|mensaje|correo)\b[^.]{0,80}(confidencia|sigilos|privilegiad)|"
    r"\b(uso exclusivo|exclusivamente|únicamente|unicamente)\b[^.]{0,30}"
    r"(destinatári|destinatari|pessoa|persona|entidade|entidad)|"
    r"\b(recebeu|recebido|receber) (esta|este) (mensagem|e-?mail)\b[^.]{0,20} por (engano|erro)|"
    r"\b(ha recibido|recibió|recibe) (este|esta) (mensaje|correo)\b[^.]{0,20} por error|"
    # the "think before printing" footer, tied to its environmental ending rather than to
    # `antes de imprimir`, which a request uses too ("antes de imprimir o boleto, confira o valor")
    r"\bantes de imprimir\b[^.]{0,100}(meio ambiente|medio ambiente|natureza|planeta|realmente necess)|"
    r"\b(meio|medio) ambiente\b[^.]{0,30}antes de imprimir|"
    # French: tied to "ce message/cet e-mail" rather than the bare word `confidentiel`,
    # which a sender's own request ("le contrat confidentiel") uses just as often
    r"\b(ce|cet|cette) (message|e-?mail|mail|courriel)\b[^.]{0,80}(confidentiel|privil[eé]gi)|"
    r"\bavez re[çc]u (ce|cet|cette) (message|e-?mail|mail)\b[^.]{0,20} par erreur|"
    r"\b(usage exclusif|exclusivement|uniquement)\b[^.]{0,30}destinataire)",
    re.I,
)
_SENTENCE = re.compile(r"(?<=[.!?])\s+")


def _starts_new_sentence(line: str) -> bool:
    """First letter is uppercase: a fresh sentence, not a wrapped line.

    Lines in uncased scripts (CJK, Devanagari, ...) never start a new piece,
    so wrapped boilerplate in those scripts still drops whole.
    """
    for ch in line:
        if ch.isalpha():
            return ch.isupper()
    return False


def _split_fused_lines(sentence: str) -> List[str]:
    """Split a fused boilerplate-positive sentence at sentence-starting newlines.

    An unpunctuated request line glued to a disclaimer line ("locked\\nThis ...")
    splits at the newline because the next line starts uppercase; a lowercase
    continuation ("are\\nconfidential") belongs to the same sentence, so a wrapped
    boilerplate footer still drops whole.
    """
    if "\n" not in sentence:
        return [sentence]
    pieces, buf = [], ""
    for line in (ln.strip() for ln in sentence.split("\n")):
        if not line:
            continue
        if buf and _starts_new_sentence(line):
            pieces.append(buf)
            buf = line
        else:
            buf = (buf + " " + line) if buf else line
    if buf:
        pieces.append(buf)
    return pieces


def _strip_disclaimer(paragraph: str) -> str:
    """Drop boilerplate disclaimer text from one paragraph.

    A paragraph is dropped whole only when *every* sentence in it is boilerplate; otherwise only
    the boilerplate sentences go. A footer that runs on without a blank line used to take the
    sender's actual request with it, which is worse than leaving one boilerplate line behind.
    """
    if not _DISCLAIMER.search(paragraph):
        return paragraph  # nothing to do: keep the original line structure
    parts = [p.strip() for p in _SENTENCE.split(paragraph) if p.strip()]
    pieces = []
    for p in parts:
        pieces.extend(_split_fused_lines(p) if _DISCLAIMER.search(p) else [p])
    return " ".join(p for p in pieces if not _DISCLAIMER.search(p))


def clean_email_body(body: str, max_chars: int = 3000) -> str:
    """Remove quoted email history, signatures and disclaimers to keep input focused.

    `max_chars` is the length the result is cut to, 3000 characters unless raised -- see
    `email_state`, which takes the same budget and passes it through.
    """
    text = (body or "").replace("\r\n", "\n").replace("\r", "\n").replace("\\n", "\n")
    # Bound regex work before the expensive patterns below: _DISCLAIMER uses
    # [^.]{0,60/80/100} alternations whose cost grows with input length, and only
    # max_chars are ever returned. Truncate lines too so one MB-long line cannot
    # dominate matching.
    if len(text) > max_chars * 4:
        text = text[: max_chars * 4]
    lines = []
    src = text.split("\n")
    for i, line in enumerate(src):
        if any(p.match(line) for p in _QUOTE_HEADERS) and lines:
            break
        if (
            lines
            and _HEADER_FROM_NAME.match(line)
            and i + 1 < len(src)
            and _HEADER_NEXT.match(src[i + 1])
        ):
            break
        if _ATTRIBUTION_TAIL.match(line) and lines:
            if _ATTRIBUTION_HEAD.match(lines[-1]):
                lines.pop()
            break
        if line.lstrip().startswith(">"):
            continue
        lines.append(line.rstrip())
    cut = len(lines)
    for i in range(max(1, min(int(len(lines) * 0.6), len(lines) - 8)), len(lines)):
        n = len(lines[i].strip())
        if (n <= 40 and any(_marker_matches(p, lines[i]) for p in _SIGNATURE_MARKERS)) or (
            n <= 60 and _DEVICE_FOOTER.match(lines[i])
        ):
            cut = i
            break
    lines = lines[:cut]
    paragraphs = [_strip_disclaimer(p) for p in re.split(r"\n\s*\n", "\n".join(lines))]
    text = re.sub(r"[ \t]+", " ", "\n\n".join(p.strip() for p in paragraphs if p.strip()))
    return text[:max_chars]


def email_state(
    subject: str,
    body: str,
    sender: Optional[str] = None,
    clean: bool = True,
    max_chars: int = 3000,
    **extra,
) -> Dict:
    """Construct a clean state dictionary for email classification.

    `max_chars` is the budget `clean_email_body` cuts the body to, and it is worth raising for a
    long message: at the default the body stops after 3000 characters, so a request that arrives in
    the last paragraphs never reaches the model -- including through `predict_long`, which scans a
    state in windows precisely so it can read past one window's worth. Ignored when `clean=False`,
    which passes the body through whole.

    Any other keyword becomes a field of the state, so it is read by the model; a typo here is an
    input mutation, not an error. The exception is a keyword whose value is exactly ``None``: it is
    dropped rather than added, so ``email_state(..., thread_id=None)`` gives the same state as
    leaving the keyword out. The key is absent, not null -- and the distinction reaches the model,
    because `laya.common.serialize_state` turns the state into JSON before it is tokenized, so a
    retained ``None`` would put ``"thread_id": null`` in the prompt. ``sender`` is dropped on the
    same call by a different rule, falsiness rather than ``None``, so ``sender=""`` leaves no
    ``from`` field either.
    """
    state = {
        "subject": (subject or "").strip(),
        "body": clean_email_body(body, max_chars=max_chars) if clean else (body or ""),
    }
    if sender:
        state["from"] = sender
    state.update({k: v for k, v in extra.items() if v is not None})
    return state
