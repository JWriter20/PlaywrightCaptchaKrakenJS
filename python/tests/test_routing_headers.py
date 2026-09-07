"""Tests for the fleet-routing and hosted-API request headers.

`routing_headers` turns CAPTCHA_REQUEST_PRIORITY into the `X-JH-Priority` header
that the fleet's haproxy front routes on (values >5 → backup GPUs). The whole
point is that it fires ONLY when deliberately set — an unset or malformed value
must never silently tag production traffic for the backups — so that boundary is
what these pin. Hermetic: no server, no network.

It also emits the three hosted-API headers: `X-CK-Client` (which integration
made the request, used to attribute camoufox revenue), `X-CK-Session` (groups
the rounds of one captcha into a single billable attempt) and `X-CK-Site` (the
hostname the solve happened on, which is what lets a solve rate be read per
site). The first two carry money implications and the third carries a
customer's, so the tests below pin that all three are absent unless set, survive
independently of a malformed priority, and can never inject extra headers.
"""
from captchakraken.planner import (
    _CLIENT_HEADER,
    _PRIORITY_HEADER,
    _REPORT_SITE_ENV,
    _SESSION_HEADER,
    _SITE_HEADER,
    routing_headers,
)


def test_no_priority_env_yields_no_header():
    assert routing_headers(env={}) == {}


def test_empty_or_whitespace_value_yields_no_header():
    assert routing_headers(env={"CAPTCHA_REQUEST_PRIORITY": ""}) == {}
    assert routing_headers(env={"CAPTCHA_REQUEST_PRIORITY": "   "}) == {}


def test_non_integer_value_is_ignored_not_forwarded():
    # A typo must not tag traffic — better no routing than wrong routing.
    assert routing_headers(env={"CAPTCHA_REQUEST_PRIORITY": "low"}) == {}
    assert routing_headers(env={"CAPTCHA_REQUEST_PRIORITY": "5.5"}) == {}


def test_integer_value_becomes_the_header():
    assert routing_headers(env={"CAPTCHA_REQUEST_PRIORITY": "10"}) == {_PRIORITY_HEADER: "10"}


def test_value_is_normalized_to_a_bare_int_string():
    # haproxy compares it as an integer (req.hdr_val), so surrounding whitespace
    # or a leading zero must not reach the wire as-is.
    assert routing_headers(env={"CAPTCHA_REQUEST_PRIORITY": " 07 "}) == {_PRIORITY_HEADER: "7"}


def test_the_tier2_default_of_10_clears_the_routing_threshold():
    # tier2_gate.sh defaults CAPTCHA_REQUEST_PRIORITY=10; haproxy routes >5 to the
    # backups, so 10 must survive as an int well above the threshold.
    hdr = routing_headers(env={"CAPTCHA_REQUEST_PRIORITY": "10"})
    assert int(hdr[_PRIORITY_HEADER]) > 5


# ── Hosted-API headers ──────────────────────────────────────────────────────


def test_self_hosted_users_send_none_of_the_hosted_headers():
    # The default path must stay byte-identical for self-hosters. Every one of
    # these is set by the driver per solve, so an empty environment is what a
    # self-hosted vLLM sees — no attribution, no session, no site.
    assert routing_headers(env={}) == {}


def test_client_and_session_are_forwarded_when_set():
    hdrs = routing_headers(
        env={
            "CAPTCHA_KRAKEN_CLIENT": "camoufox/0.4.11",
            "CAPTCHA_KRAKEN_SESSION": "6f1a2b3c-0000-4000-8000-000000000001",
        }
    )
    assert hdrs[_CLIENT_HEADER] == "camoufox/0.4.11"
    assert hdrs[_SESSION_HEADER] == "6f1a2b3c-0000-4000-8000-000000000001"


def test_the_site_is_forwarded_when_set():
    hdrs = routing_headers(env={"CAPTCHA_KRAKEN_SITE": "checkout.example.com"})
    assert hdrs == {_SITE_HEADER: "checkout.example.com"}


def test_the_site_can_be_switched_off_on_its_own():
    """A SEPARATE disclosure from the outcome report, so a separate switch.

    The outcome is a fact about our model — did it get this one right — and it
    is what makes a failed board improvable at all. The site is a fact about the
    customer's business. An account can reasonably want the first sent and not
    the second, and one switch for both would price that choice at "tell us
    nothing", which the failure corpus is what loses.
    """
    env = {"CAPTCHA_KRAKEN_SITE": "checkout.example.com",
           "CAPTCHA_KRAKEN_SESSION": "6f1a2b3c-0000-4000-8000-000000000001"}
    assert routing_headers(env={**env, _REPORT_SITE_ENV: "0"}) == {
        _SESSION_HEADER: "6f1a2b3c-0000-4000-8000-000000000001"}
    # ...and only "0". An unset or unrecognised value keeps the default, exactly
    # as CAPTCHA_REPORT_OUTCOME does, so a typo cannot quietly stop collection.
    assert _SITE_HEADER in routing_headers(env={**env, _REPORT_SITE_ENV: "no"})
    assert _SITE_HEADER in routing_headers(env=env)


def test_switching_the_site_off_does_not_take_the_others_with_it():
    # The same independence the malformed-priority test below pins, on the one
    # header a caller can disable. A switch that suppressed the session id would
    # break the per-attempt billing cap for anyone who used it.
    hdrs = routing_headers(env={
        _REPORT_SITE_ENV: "0",
        "CAPTCHA_KRAKEN_SITE": "checkout.example.com",
        "CAPTCHA_KRAKEN_CLIENT": "camoufox/0.4.11",
        "CAPTCHA_REQUEST_PRIORITY": "10",
    })
    assert hdrs == {_CLIENT_HEADER: "camoufox/0.4.11", _PRIORITY_HEADER: "10"}


def test_blank_values_are_dropped_rather_than_sent_empty():
    # An empty header would read as "attributed to nothing" downstream; absent is
    # unambiguous.
    assert routing_headers(env={"CAPTCHA_KRAKEN_CLIENT": "   "}) == {}
    assert routing_headers(env={"CAPTCHA_KRAKEN_SESSION": ""}) == {}


def test_a_malformed_priority_does_not_suppress_attribution():
    # Regression guard: the headers are derived independently. If a typo'd
    # priority swallowed the client header, camoufox traffic would silently be
    # counted as direct and understate the partner's revenue share.
    hdrs = routing_headers(
        env={"CAPTCHA_REQUEST_PRIORITY": "oops", "CAPTCHA_KRAKEN_CLIENT": "camoufox/1.0"}
    )
    assert hdrs == {_CLIENT_HEADER: "camoufox/1.0"}


def test_crlf_cannot_inject_additional_headers():
    # These values come from the environment and reach the wire verbatim, so a
    # newline must never be able to splice in another header.
    hdrs = routing_headers(
        env={"CAPTCHA_KRAKEN_CLIENT": "camoufox\r\nX-CK-Session: forged"}
    )
    assert hdrs == {_CLIENT_HEADER: "camoufoxX-CK-Session: forged"}
    assert "\r" not in hdrs[_CLIENT_HEADER] and "\n" not in hdrs[_CLIENT_HEADER]
    assert _SESSION_HEADER not in hdrs


def test_oversized_value_is_truncated_not_dropped():
    hdrs = routing_headers(env={"CAPTCHA_KRAKEN_CLIENT": "c" * 500})
    assert len(hdrs[_CLIENT_HEADER]) == 128


# ── CAPTCHA_KRAKEN_EXTRA_HEADERS ────────────────────────────────────────────
#
# The escape hatch for an endpoint gated by something that is not the API key —
# our own dev gateway wants X-CK-Dev-Auth, and a corporate egress proxy may want
# its own token. These assert the two properties that matter: malformed input is
# dropped rather than guessed at, and the credential/attribution headers cannot
# be overwritten from the environment.

from captchakraken.planner import _EXTRA_HEADERS_ENV  # noqa: E402


def _extra(value):
    return routing_headers(env={_EXTRA_HEADERS_ENV: value})


def test_absent_and_empty_add_nothing():
    assert routing_headers(env={}) == {}
    assert _extra("") == {}
    assert _extra("   \n  ") == {}


def test_single_pair():
    assert _extra("X-CK-Dev-Auth: hunter2") == {"X-CK-Dev-Auth": "hunter2"}


def test_whitespace_around_the_pair_is_ignored():
    assert _extra("  X-CK-Dev-Auth :  hunter2  ") == {"X-CK-Dev-Auth": "hunter2"}


def test_several_pairs_by_newline_and_comma():
    assert _extra("X-One: a\nX-Two: b") == {"X-One": "a", "X-Two": "b"}
    assert _extra("X-One: a, X-Two: b") == {"X-One": "a", "X-Two": "b"}


def test_a_value_may_contain_a_colon():
    # partition() splits on the FIRST colon, so a bearer-ish value survives.
    assert _extra("X-Proxy: Basic abc:def") == {"X-Proxy": "Basic abc:def"}


def test_entries_without_a_colon_are_dropped_not_guessed():
    assert _extra("nonsense") == {}
    assert _extra("X-Good: yes\nnonsense") == {"X-Good": "yes"}


def test_empty_name_or_value_is_dropped():
    assert _extra(": value") == {}
    assert _extra("X-Empty:") == {}
    assert _extra("X-Empty:    ") == {}


def test_crlf_cannot_splice_another_header():
    # The value sanitizer strips anything outside printable ASCII, so a CR/LF
    # in the value cannot start a new header line on the wire.
    out = _extra("X-Bad: a\r\nX-Injected: b")
    assert out == {"X-Bad": "a", "X-Injected": "b"}  # split on the newline, not spliced
    for value in out.values():
        assert "\r" not in value and "\n" not in value


def test_invalid_header_names_are_dropped():
    assert _extra("Bad Name: v") == {}
    assert _extra("X-Bad\tName: v") == {}


def test_protected_headers_cannot_be_overwritten():
    # The credential and the billing attribution are not env-overridable. A
    # stray export must never be able to change which key is charged, or pin
    # one session id and escape the per-attempt billing cap.
    assert _extra("Authorization: Bearer stolen") == {}
    assert _extra("authorization: Bearer stolen") == {}
    assert _extra("Content-Type: text/plain") == {}
    assert _extra("X-CK-Client: not-camoufox") == {}
    assert _extra("X-CK-Session: pinned") == {}
    # The site too: an env var that could rewrite it would let one account's
    # traffic be filed under another's domain, and the per-site rate is read as
    # if the client had measured it.
    assert _extra("X-CK-Site: someone-elses.example.com") == {}


def test_protected_names_do_not_block_the_rest_of_the_line():
    assert _extra("Authorization: nope\nX-Fine: yes") == {"X-Fine": "yes"}


def test_extra_headers_compose_with_the_others():
    out = routing_headers(
        env={
            "CAPTCHA_KRAKEN_CLIENT": "camoufox/0.4.11",
            _EXTRA_HEADERS_ENV: "X-CK-Dev-Auth: hunter2",
        }
    )
    assert out == {"X-CK-Client": "camoufox/0.4.11", "X-CK-Dev-Auth": "hunter2"}
