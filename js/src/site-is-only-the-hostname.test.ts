/**
 * What `X-CK-Site` may contain, decided at the one place that reads the page.
 *
 * A captcha appears on a login page, a checkout, a password reset — precisely
 * the set of URLs whose PATH AND QUERY you would least want sent anywhere.
 * `page.url()` is the full thing and the header is one string; `siteOf` is the
 * only place that difference is decided.
 *
 * Twin of `python/tests/test_only_the_hostname_leaves_the_machine.py`, and the
 * parity is not decorative here: this port reaches the API through that Python
 * CLI, so a hostname this side got wrong arrives already wrong and nothing
 * downstream re-derives it.
 */

import { test } from 'node:test';
import assert from 'node:assert/strict';

import { siteOf } from './solver';

/** Playwright and Puppeteer both expose `url()`; some adapters use a property. */
const asMethod = (url: unknown) => ({ url: () => url }) as never;
const asProperty = (url: unknown) => ({ url }) as never;

test('the hostname, from either page shape', () => {
  assert.equal(siteOf(asMethod('https://checkout.example.com/cart')), 'checkout.example.com');
  assert.equal(siteOf(asProperty('https://checkout.example.com/cart')), 'checkout.example.com');
});

test('everything after the host is dropped', () => {
  // Each of these is a real shape a captcha sits behind, and each carries
  // something in the part that does not survive.
  const cases: [string, string][] = [
    ['https://shop.example.com/account/reset?token=9f3c1a', 'shop.example.com'],
    ['https://shop.example.com/orders/8812/invoice', 'shop.example.com'],
    ['https://user:hunter2@shop.example.com/login', 'shop.example.com'],
    ['https://shop.example.com/login#email=a%40b.com', 'shop.example.com'],
    // The port is not sensitive, but `example.com` and `example.com:8443` are
    // one site and a rate that splits them is two halves of one number.
    ['https://staging.example.com:8443/login', 'staging.example.com'],
    ['https://Shop.Example.COM/login', 'shop.example.com'],
  ];
  for (const [url, expected] of cases) {
    assert.equal(siteOf(asMethod(url)), expected, url);
    assert.ok(!String(siteOf(asMethod(url))).includes('token'));
  }
});

test('a page with no host yields null rather than a guess', () => {
  // Absent is unambiguous, and `routing_headers` drops an empty value. This is
  // also every Tier 3 fixture run — those are served from a local port and
  // file:// URLs.
  for (const url of ['about:blank', 'file:///tmp/fixtures/recaptcha.html',
                     'data:text/html,<h1>hi</h1>', '', 'not a url at all']) {
    assert.equal(siteOf(asMethod(url)), null, url);
  }
});

test('a page that cannot be read never throws', () => {
  // `siteOf` runs at the top of solve(), before anything has been attempted, so
  // an exception here would replace a solvable captcha with an error about a
  // header.
  const exploding = { get url(): never { throw new Error('page closed'); } } as never;
  assert.equal(siteOf(exploding), null);
  assert.equal(siteOf({} as never), null);
  assert.equal(siteOf(null as never), null);
});
