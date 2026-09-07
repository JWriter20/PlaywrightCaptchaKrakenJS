"""What `X-CK-Site` may contain, decided at the one place that reads the page.

A captcha appears on a login page, a checkout, a password reset — which is
precisely the set of URLs whose PATH AND QUERY you would least want sent
anywhere. `page.url` is the full thing, and the header is one string; the only
place that difference is decided is `_site_of`, so it is where this is pinned
rather than at the header layer, which sees a value that is already whatever
this returned.

Twin of `js/src/site-is-only-the-hostname.test.ts`. Both ports must agree here,
and not approximately: the JS driver reaches the API through this same Python
CLI, so a hostname the TS side got wrong would arrive already wrong.
"""
from __future__ import annotations

import os

import pytest

from captchakraken.page_solver import _SITE_ENV, _site_of


class FakePage:
    """Playwright shape: `url` is a property returning a string."""

    def __init__(self, url):
        self.url = url


class MethodPage:
    """Puppeteer shape: `url()` is a method. Both are duck-typed, so both work."""

    def __init__(self, url):
        self._url = url

    def url(self):
        return self._url


@pytest.mark.parametrize("page_class", [FakePage, MethodPage])
def test_the_hostname_and_only_the_hostname(page_class):
    assert _site_of(page_class("https://checkout.example.com/cart")) == "checkout.example.com"


@pytest.mark.parametrize("url,expected", [
    # THE ONES THAT MATTER. Each of these is a real shape a captcha sits behind,
    # and each carries something in the part that is dropped.
    ("https://shop.example.com/account/reset?token=9f3c1a", "shop.example.com"),
    ("https://shop.example.com/orders/8812/invoice", "shop.example.com"),
    ("https://user:hunter2@shop.example.com/login", "shop.example.com"),
    ("https://shop.example.com/login#email=a%40b.com", "shop.example.com"),
    # The port is dropped too. It is not sensitive, but `example.com` and
    # `example.com:8443` are one site and a rate that splits them is two halves
    # of one number.
    ("https://staging.example.com:8443/login", "staging.example.com"),
    # Case is not identity either, for the same reason.
    ("https://Shop.Example.COM/login", "shop.example.com"),
])
def test_everything_after_the_host_is_dropped(url, expected):
    assert _site_of(FakePage(url)) == expected
    assert "token" not in _site_of(FakePage(url))


@pytest.mark.parametrize("url", [
    "about:blank",
    "file:///home/someone/fixtures/recaptcha.html",
    "data:text/html,<h1>hi</h1>",
    "",
    "not a url at all",
])
def test_a_page_with_no_host_yields_nothing_rather_than_a_guess(url):
    # Absent is unambiguous. `routing_headers` drops an empty value, so these
    # send no header at all — which is also every Tier 3 fixture run, since
    # those are served from `file://` and a local port.
    assert _site_of(FakePage(url)) == ""


def test_a_page_that_cannot_be_read_never_raises():
    """Telemetry may not fail a solve.

    `_site_of` runs at the top of `solve()`, before anything has been attempted,
    so an exception here would replace a solvable captcha with a traceback about
    a header. Every shape that could throw returns "" instead.
    """
    class Exploding:
        @property
        def url(self):
            raise RuntimeError("page has been closed")

    assert _site_of(Exploding()) == ""
    assert _site_of(object()) == ""
    assert _site_of(None) == ""


def test_the_env_var_is_the_one_planner_reads():
    # The two modules name it separately (one constant per port, per module) and
    # nothing else connects them. A rename in either would leave the header
    # silently unset — the header would just never appear, which looks exactly
    # like a customer who has switched it off.
    from captchakraken.planner import _SITE_ENV as PLANNER_SIDE

    assert _SITE_ENV == PLANNER_SIDE == "CAPTCHA_KRAKEN_SITE"


def test_a_solve_leaves_no_site_behind_it(monkeypatch):
    """The lifetime, which is the half `_site_of` cannot get right on its own.

    A stale value would file the NEXT captcha under this page's domain. That is
    worse than filing it under none: a per-site rate is only worth reading if
    every row in it is a site the solve actually happened on, and there is
    nothing downstream that could notice one that is not.
    """
    monkeypatch.delenv(_SITE_ENV, raising=False)
    from captchakraken import page_solver as PS

    solver = PS.PageSolver()
    seen = {}

    # Drive `solve`'s env handling without a browser: the body raises, which is
    # also the case that matters most — a solve that throws must still clean up.
    def boom(*a, **k):
        seen["site"] = os.environ.get(_SITE_ENV)
        raise RuntimeError("no browser here")

    monkeypatch.setattr(PS.PageSolver, "_solve_impl", boom)
    with pytest.raises(RuntimeError):
        solver.solve(FakePage("https://shop.example.com/login"))

    assert seen["site"] == "shop.example.com", "the solve did not see its own site"
    assert _SITE_ENV not in os.environ, "the site outlived the solve that set it"
