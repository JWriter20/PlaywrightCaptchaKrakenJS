/**
 * The tools.
 *
 * WHAT THIS SERVER IS FOR: letting an agent provision and watch its own captcha
 * solving. Sign the human in, mint a key for the hosted endpoint, report what has
 * been spent, and hand over a payment link when the balance runs low. It does
 * not solve captchas — that is the gateway's OpenAI-compatible endpoint, and
 * the key minted here is what talks to it.
 *
 * TWO RULES SHAPE EVERY TOOL BELOW.
 *
 *  1. NOTHING SPENDS MONEY WITHOUT A HUMAN. `get_topup_link` returns a URL. It
 *     does not charge a card, and there is no tool that can. An agent may
 *     *offer* to spend; a person clicks.
 *
 *  2. NO TOOL OUTPUT EVER CONTAINS A LIVE CREDENTIAL. `create_api_key` writes
 *     the minted key straight to a 0600 file and reports the PATH; the solver
 *     reads it from there. Returning the secret — which this tool used to do —
 *     puts it in the transcript the moment an agent repeats it in a summary,
 *     and asking the agent nicely not to is not a control. The management
 *     token obtained by `sign_in` is likewise written to disk and never
 *     printed.
 *
 * Read-only tools are annotated as such so a client can decide what needs
 * confirming. `create_api_key` and `revoke_api_key` are not read-only and are
 * marked destructive where they are.
 */

import { McpServer } from '@modelcontextprotocol/sdk/server/mcp.js';
import { z } from 'zod';

import { ApiError, ControlPlane } from './api.js';
import {
  clearCredential,
  credentialPath,
  hasLiveToken,
  loadCredential,
  saveCredential,
  solverCredentialPath,
  writeSolverCredential,
} from './credentials.js';
import type { PendingDevice } from './credentials.js';

/**
 * How long `sign_in` waits before handing control back.
 *
 * Deliberately short. MCP clients time tool calls out — 60 seconds is a common
 * default — and a sign-in that blocks for two minutes gets killed with the
 * device code stranded. So this polls for well under any of them, then returns
 * "still waiting, call me again", and the pending code is on disk so the next
 * call resumes the same flow rather than printing a second code at a human who
 * is already looking at the first one.
 */
const SIGN_IN_WAIT_MS = 25_000;

interface DeviceStart {
  device_code: string;
  user_code: string;
  verification_uri: string;
  verification_uri_complete: string;
  expires_in: number;
  interval: number;
}

interface TokenGrant {
  access_token: string;
  expires_at: string;
  account: {
    user_id: number;
    github_login: string | null;
    email: string | null;
    balance_credits: number;
    balance_usd: string;
  };
}

interface AccountResponse {
  account: {
    user_id: number;
    github_login: string | null;
    email: string | null;
    created_at: string;
    balance_credits: number;
    balance_usd: string;
    low_balance_threshold: number | null;
    low_balance: boolean;
    /**
     * Optional because a control plane older than migration 0008 does not send
     * them, and an MCP client is distributed — it meets whatever is deployed.
     * Absent reads as "a normal account", which is the safe way to be wrong.
     */
    unlimited?: boolean;
    is_admin?: boolean;
  };
  endpoint: { base_url: string; dashboard_url: string; requires_dev_header: boolean };
  credits_per_usd: number;
}

interface UsageResponse {
  window_days: number;
  totals: {
    billable_rounds: number;
    credits: number;
    usd: string;
    waived_rounds: number;
    by_client: Array<{ client: string | null; billable_rounds: number; credits: number }>;
  };
  daily: Array<{ day: string; billable_rounds: number; credits: number }>;
  recent: Array<{
    at: string;
    puzzle_class: string;
    credits: number;
    waived_reason: string | null;
    client: string | null;
    upstream_ms: number | null;
  }>;
  ledger: Array<{ at: string; kind: string; delta_credits: number; usd_amount_cents: number | null }>;
  balance_credits: number;
  balance_usd: string;
  /** Absent on a control plane older than migration 0008. */
  unlimited?: boolean;
}

interface KeyListResponse {
  max_active_keys: number;
  keys: Array<{
    id: number;
    name: string | null;
    masked: string;
    created_at: string;
    last_used_at: string | null;
    revoked: boolean;
  }>;
}

interface CreatedKeyResponse {
  id: number;
  api_key: string;
  masked: string;
  name: string | null;
  base_url: string;
}

type ToolResult = {
  content: Array<{ type: 'text'; text: string }>;
  isError?: boolean;
};

function text(body: string): ToolResult {
  return { content: [{ type: 'text', text: body }] };
}

function failure(body: string): ToolResult {
  return { content: [{ type: 'text', text: body }], isError: true };
}

/**
 * Turn a thrown ApiError into something an agent can act on.
 *
 * `not_signed_in` gets the instruction rather than the raw message, because it
 * is the one failure with an obvious next step and an agent that is told the
 * step will take it instead of reporting an error to the human.
 */
function describe(error: unknown): ToolResult {
  if (error instanceof ApiError) {
    if (error.code === 'not_signed_in' || error.status === 401) {
      return failure('Not signed in to CaptchaKraken. Run the `sign_in` tool first.');
    }
    return failure(`${error.code}: ${error.message}`);
  }
  return failure(error instanceof Error ? error.message : String(error));
}

const sleep = (ms: number) => new Promise((resolve) => setTimeout(resolve, ms));

export function createServer(baseUrl: string, clientName: string): McpServer {
  const api = new ControlPlane(baseUrl);
  const server = new McpServer(
    { name: 'captchakraken', version: '0.1.3' },
    {
      instructions:
        'CaptchaKraken account management. Use sign_in once to connect a GitHub account, ' +
        'create_api_key to get a key for the captcha-solving endpoint, and get_usage / ' +
        'get_balance to see what has been spent. This server does not solve captchas itself — ' +
        'the key it mints is used against the OpenAI-compatible endpoint reported by get_account.',
    },
  );

  // ── signing in ────────────────────────────────────────────────────────────

  server.registerTool(
    'sign_in',
    {
      title: 'Sign in to CaptchaKraken',
      description:
        'Connect a CaptchaKraken account using GitHub, creating one if it does not exist. ' +
        'Prints a short code and a link for the human to approve in a browser, then waits ' +
        'briefly. If it returns still-waiting, call it again to resume the same sign-in — it ' +
        'will not start a second one. New accounts get free trial credits and a first API key.',
      inputSchema: {
        client_name: z
          .string()
          .optional()
          .describe('Name shown on the approval page, e.g. the editor this is running in.'),
      },
      annotations: { title: 'Sign in to CaptchaKraken', openWorldHint: true },
    },
    async ({ client_name }) => {
      try {
        const credential = loadCredential(baseUrl);

        if (hasLiveToken(credential)) {
          const who = credential.account?.githubLogin ?? `account ${credential.account?.userId}`;
          return text(
            `Already signed in as ${who} on ${baseUrl}.\n` +
              'Run `sign_out` first if you want to connect a different account.',
          );
        }

        // Resume a code that is still live rather than printing a second one at
        // a human who is already looking at the first.
        let pending = credential.pending;
        if (!pending || pending.expiresAtMs <= Date.now()) {
          const started = await api.request<DeviceStart>('/api/v1/device/start', {
            method: 'POST',
            authenticated: false,
            body: { client_name: client_name ?? clientName },
          });
          pending = {
            deviceCode: started.device_code,
            userCode: started.user_code,
            verificationUri: started.verification_uri,
            verificationUriComplete: started.verification_uri_complete,
            expiresAtMs: Date.now() + started.expires_in * 1000,
            intervalSeconds: started.interval,
          };
          saveCredential(baseUrl, { ...credential, pending });
        }

        const outcome = await pollForToken(api, pending);

        if (outcome.kind === 'granted') {
          saveCredential(baseUrl, {
            accessToken: outcome.grant.access_token,
            expiresAt: outcome.grant.expires_at,
            account: {
              userId: outcome.grant.account.user_id,
              githubLogin: outcome.grant.account.github_login,
              email: outcome.grant.account.email,
            },
          });
          const who = outcome.grant.account.github_login ?? `account ${outcome.grant.account.user_id}`;
          return text(
            `Signed in as ${who}.\n` +
              `Balance: ${outcome.grant.account.balance_usd} (${outcome.grant.account.balance_credits.toLocaleString('en-US')} credits).\n` +
              `The token is stored at ${credentialPath()} and can be revoked from the dashboard at any time.\n\n` +
              'Next: `create_api_key` mints a key for the solving endpoint.',
          );
        }

        if (outcome.kind === 'waiting') {
          return text(
            `Waiting for approval.\n\n` +
              `  Open:  ${pending.verificationUriComplete}\n` +
              `  Code:  ${pending.userCode}\n\n` +
              'Ask the human to open that link and approve the code. Then call `sign_in` again ' +
              'to resume — it will pick up this same request, not start a new one.',
          );
        }

        // Denied or dead. Drop the pending code so the next call starts clean.
        const { pending: _dropped, ...withoutPending } = credential;
        saveCredential(baseUrl, withoutPending);
        return failure(
          outcome.kind === 'denied'
            ? 'The sign-in was declined in the browser. Nothing was connected.'
            : 'That sign-in code expired or was already used. Call `sign_in` again for a new one.',
        );
      } catch (error) {
        return describe(error);
      }
    },
  );

  server.registerTool(
    'sign_out',
    {
      title: 'Sign out',
      description:
        'Revoke this client’s access token on the server and delete it from disk. API keys ' +
        'already minted keep working — this disconnects the management client, not the solving.',
      annotations: { title: 'Sign out', destructiveHint: true, openWorldHint: true },
    },
    async () => {
      try {
        // Revoke server-side first. If that fails we still have the token, and
        // deleting it locally would leave a live credential nobody can reach to
        // revoke — the file is the only copy of it.
        await api.request('/api/v1/signout', { method: 'POST' });
        clearCredential(baseUrl);
        return text('Signed out. The token has been revoked on the server and removed from disk.');
      } catch (error) {
        if (error instanceof ApiError && (error.status === 401 || error.code === 'not_signed_in')) {
          clearCredential(baseUrl);
          return text('There was no live token. Local state cleared.');
        }
        return describe(error);
      }
    },
  );

  // ── the account ───────────────────────────────────────────────────────────

  server.registerTool(
    'get_account',
    {
      title: 'Account and endpoint',
      description:
        'Who is signed in, the credit balance, and the base URL to point an OpenAI-compatible ' +
        'client at for solving.',
      annotations: { title: 'Account and endpoint', readOnlyHint: true, openWorldHint: true },
    },
    async () => {
      try {
        const me = await api.request<AccountResponse>('/api/v1/me');
        const lines = [
          `Account:   ${me.account.github_login ?? `#${me.account.user_id}`}${me.account.is_admin ? ' (admin)' : ''}`,
          `Email:     ${me.account.email ?? '(none on file)'}`,
          // An exempt account's balance is real and simply never spent. Printing
          // it alone would have an agent reason about a number that cannot move,
          // and offer top-ups against it.
          me.account.unlimited
            ? `Balance:   unlimited — this account is not billed for solves`
            : `Balance:   ${me.account.balance_usd} (${me.account.balance_credits.toLocaleString('en-US')} credits)`,
          `Endpoint:  ${me.endpoint.base_url}`,
          `Dashboard: ${me.endpoint.dashboard_url}`,
        ];
        if (me.account.low_balance) {
          lines.push('', 'Balance is low. `get_topup_link` returns a payment link for the human.');
        }
        if (me.endpoint.requires_dev_header) {
          lines.push(
            '',
            'This is a development deployment: its gateway also requires the shared X-CK-Dev-Auth header in front of the key.',
          );
        }
        return text(lines.join('\n'));
      } catch (error) {
        return describe(error);
      }
    },
  );

  server.registerTool(
    'get_balance',
    {
      title: 'Credit balance',
      description: 'The remaining credit balance, in credits and in dollars. Nothing else.',
      annotations: { title: 'Credit balance', readOnlyHint: true, openWorldHint: true },
    },
    async () => {
      try {
        const me = await api.request<AccountResponse>('/api/v1/me');
        if (me.account.unlimited) {
          return text('unlimited — this account is not billed for solves');
        }
        return text(
          `${me.account.balance_usd} (${me.account.balance_credits.toLocaleString('en-US')} credits)` +
            (me.account.low_balance ? ' — low' : ''),
        );
      } catch (error) {
        return describe(error);
      }
    },
  );

  server.registerTool(
    'get_usage',
    {
      title: 'Usage and spending',
      description:
        'Billable inference responses and credits spent, per day and in total, plus recent ' +
        'purchases. The server keeps a rolling 30-day window; `days` narrows what is reported.',
      inputSchema: {
        days: z
          .number()
          .int()
          .min(1)
          .max(30)
          .optional()
          .describe('How many of the last 30 days to report. Defaults to all 30.'),
      },
      annotations: { title: 'Usage and spending', readOnlyHint: true, openWorldHint: true },
    },
    async ({ days }) => {
      try {
        const usage = await api.request<UsageResponse>('/api/v1/usage');

        // Sliced here rather than asked for: the window is fixed server-side
        // because that is what makes the read cheap. See the note on the route.
        const window = days ?? usage.window_days;
        const daily = usage.daily.slice(-window);
        const credits = daily.reduce((sum, day) => sum + day.credits, 0);
        const rounds = daily.reduce((sum, day) => sum + day.billable_rounds, 0);

        const lines = [
          `Last ${window} day${window === 1 ? '' : 's'}:`,
          `  ${rounds.toLocaleString('en-US')} billable responses`,
          `  ${credits.toLocaleString('en-US')} credits spent`,
          `Balance now: ${usage.balance_usd} (${usage.balance_credits.toLocaleString('en-US')} credits)`,
        ];

        if (usage.unlimited) {
          lines.push(
            '',
            'This account is billing-exempt: every response above was served free, and the balance does not move.',
          );
        } else if (usage.totals.waived_rounds > 0) {
          lines.push(
            '',
            `${usage.totals.waived_rounds} response(s) in the full window were served free — our own retries and the per-attempt cap. Those are not billing errors.`,
          );
        }

        if (usage.totals.by_client.length > 1) {
          lines.push('', 'By integration (full window):');
          for (const row of usage.totals.by_client) {
            lines.push(
              `  ${row.client ?? 'direct'}: ${row.billable_rounds.toLocaleString('en-US')} rounds, ${row.credits.toLocaleString('en-US')} credits`,
            );
          }
        }

        const spending = daily.filter((day) => day.credits > 0);
        if (spending.length > 0) {
          lines.push('', 'Per day:');
          for (const day of spending) {
            lines.push(`  ${day.day}  ${day.credits.toLocaleString('en-US')} credits`);
          }
        }

        if (usage.ledger.length > 0) {
          lines.push('', 'Purchases and grants:');
          for (const entry of usage.ledger.slice(0, 10)) {
            const paid =
              entry.usd_amount_cents === null ? '' : ` ($${(entry.usd_amount_cents / 100).toFixed(2)})`;
            lines.push(
              `  ${entry.at.slice(0, 10)}  ${entry.kind}  ${entry.delta_credits > 0 ? '+' : ''}${entry.delta_credits.toLocaleString('en-US')}${paid}`,
            );
          }
        }

        return text(lines.join('\n'));
      } catch (error) {
        return describe(error);
      }
    },
  );

  // ── keys ──────────────────────────────────────────────────────────────────

  server.registerTool(
    'list_api_keys',
    {
      title: 'List API keys',
      description:
        'The account’s solving keys, masked. The secret half is not stored anywhere and cannot ' +
        'be listed — only minted.',
      annotations: { title: 'List API keys', readOnlyHint: true, openWorldHint: true },
    },
    async () => {
      try {
        const listing = await api.request<KeyListResponse>('/api/v1/keys');
        if (listing.keys.length === 0) {
          return text('No keys yet. `create_api_key` mints one.');
        }
        const rows = listing.keys.map((key) => {
          const state = key.revoked ? 'revoked' : 'live';
          const used = key.last_used_at ? `last used ${key.last_used_at.slice(0, 10)}` : 'never used';
          return `  #${key.id}  ${key.masked}  ${key.name ?? '(unnamed)'}  [${state}, ${used}]`;
        });
        const live = listing.keys.filter((key) => !key.revoked).length;
        return text(
          [`${live} live of ${listing.max_active_keys} allowed:`, ...rows].join('\n'),
        );
      } catch (error) {
        return describe(error);
      }
    },
  );

  server.registerTool(
    'create_api_key',
    {
      title: 'Create an API key',
      description:
        'Mint a key for the captcha-solving endpoint and save it to disk where ' +
        'the solver will find it. THE SECRET IS NEVER RETURNED — it is written straight to a ' +
        '0600 file and only the path is reported back, so it cannot leak through this ' +
        'conversation. After this runs, a solve works with no environment variables set.',
      inputSchema: {
        name: z
          .string()
          .max(60)
          .optional()
          .describe('A label so this key is identifiable in the dashboard, e.g. "scraper-prod".'),
      },
      annotations: { title: 'Create an API key', readOnlyHint: false, openWorldHint: true },
    },
    async ({ name }) => {
      try {
        const created = await api.request<CreatedKeyResponse>('/api/v1/keys', {
          method: 'POST',
          body: { name: name ?? null },
        });

        // The secret goes to disk and NOT into this tool's result. Everything
        // returned below is safe to repeat: an id, a masked form, and a path.
        //
        // The write is the last thing that can fail, and if it does the key
        // already exists server-side — so say so plainly rather than pretending
        // nothing happened, and point at revoke. Silently swallowing this would
        // strand a live key that the user does not know they own.
        let credentialFile: string;
        try {
          credentialFile = writeSolverCredential({
            apiKey: created.api_key,
            baseUrl: created.base_url,
          });
        } catch (writeError) {
          const reason = writeError instanceof Error ? writeError.message : String(writeError);
          return text(
            [
              `Key #${created.id} was created (${created.masked}), but saving it to ` +
                `${solverCredentialPath()} failed: ${reason}`,
              '',
              'The secret is NOT being printed here — it would end up in this transcript.',
              'The key exists server-side and cannot be retrieved again, so either fix the',
              `permissions on that path and mint a new key, or revoke key #${created.id} with`,
              'revoke_api_key so nothing untracked is left live.',
            ].join('\n'),
          );
        }

        return text(
          [
            `Key #${created.id} created${created.name ? ` (${created.name})` : ''}, shown as ${created.masked} in the dashboard.`,
            '',
            `Saved to ${credentialFile} (0600), pointing at ${created.base_url}.`,
            '',
            'The secret itself is deliberately not shown — it went straight to that file so it',
            'never enters this conversation, and only a SHA-256 is stored server-side. Solving',
            'now works with no environment variables set at all.',
          ].join('\n'),
        );
      } catch (error) {
        return describe(error);
      }
    },
  );

  server.registerTool(
    'revoke_api_key',
    {
      title: 'Revoke an API key',
      description:
        'Kill a key by its id. It stops working within about 30 seconds — the gateway’s refresh ' +
        'interval. This cannot be undone; mint a new key instead of trying to restore one.',
      inputSchema: {
        id: z.number().int().positive().describe('The key id from list_api_keys.'),
      },
      annotations: { title: 'Revoke an API key', readOnlyHint: false, destructiveHint: true, openWorldHint: true },
    },
    async ({ id }) => {
      try {
        await api.request(`/api/v1/keys/${id}/revoke`, { method: 'POST' });
        return text(`Key #${id} revoked. It stops working within 30 seconds.`);
      } catch (error) {
        return describe(error);
      }
    },
  );

  // ── money ─────────────────────────────────────────────────────────────────

  server.registerTool(
    'get_topup_link',
    {
      title: 'Get a top-up link',
      description:
        'A URL the human opens to add credits. THIS DOES NOT CHARGE ANYTHING — it returns a ' +
        'link; a person completes the payment on Stripe. Any whole dollar amount from $5 is ' +
        'accepted, and larger amounts earn bonus credits. Omit `usd` to let them choose the ' +
        'amount, which is the right default when the agent has not been told what to spend.',
      inputSchema: {
        usd: z
          .number()
          .int()
          .min(5)
          .optional()
          .describe(
            'A whole dollar amount, $5 or more. Larger amounts earn bonus credits. Omit to ' +
              'return the chooser page, where the human picks the amount themselves.',
          ),
      },
      annotations: { title: 'Get a top-up link', readOnlyHint: false, openWorldHint: true },
    },
    async ({ usd }) => {
      try {
        const result = await api.request<{
          url: string;
          kind: 'chooser' | 'checkout';
          usd?: number;
          credits?: number;
          packs?: Array<{ usd: number; credits: number; bonus_percent?: number }>;
          min_usd?: number;
          max_usd?: number;
        }>('/api/v1/billing/checkout', {
          method: 'POST',
          body: usd === undefined ? {} : { usd },
        });

        if (result.kind === 'checkout') {
          return text(
            `Stripe checkout for $${result.usd} (${result.credits?.toLocaleString('en-US')} credits):\n\n  ${result.url}\n\n` +
              'Nothing has been charged. The human completes the payment on that page, and the credits land within seconds of it succeeding.',
          );
        }

        // The packs are SUGGESTIONS now, not the set of amounts on sale — the
        // page takes any whole dollar amount in the range. Presenting them as
        // "available packs" is what would make an agent tell a human they have
        // to pick one of three.
        const packs = (result.packs ?? [])
          .map(
            (pack) =>
              `  $${pack.usd} → ${pack.credits.toLocaleString('en-US')} credits` +
              (pack.bonus_percent ? ` (+${pack.bonus_percent}% bonus)` : ''),
          )
          .join('\n');
        const range =
          result.min_usd && result.max_usd
            ? `Any whole dollar amount from $${result.min_usd} to $${result.max_usd}.`
            : 'Any whole dollar amount.';
        return text(
          `Top-up page:\n\n  ${result.url}\n\n${range} Common choices:\n${packs}\n\nNothing has been charged.`,
        );
      } catch (error) {
        return describe(error);
      }
    },
  );

  server.registerTool(
    'get_pricing',
    {
      title: 'Pricing',
      description:
        'The rate card: dollars per 1,000 inference responses for images and for video, how many ' +
        'responses a captcha typically takes, and the per-captcha billing ceiling. Use this to ' +
        'estimate a job before running it.',
      annotations: { title: 'Pricing', readOnlyHint: true, openWorldHint: true },
    },
    async () => {
      try {
        const pricing = await api.request<{
          credits_per_usd: number;
          max_billable_responses_per_session: number;
          max_responses_per_session: number;
          free_challenges: string[];
          classes: Array<{
            puzzle_class: string;
            label: string;
            covers: string;
            credits_per_response: number;
            usd_per_1000_responses: number;
            typical_responses_per_captcha: string;
          }>;
          credit_packs: Array<{ usd: number; credits: number }>;
        }>('/api/v1/pricing', { authenticated: false });

        const lines = [
          `Billed per inference response. $1.00 = ${pricing.credits_per_usd.toLocaleString('en-US')} credits.`,
          '',
        ];
        for (const row of pricing.classes) {
          lines.push(
            `${row.label}: $${row.usd_per_1000_responses.toFixed(2)} per 1,000 (${row.credits_per_response} credits each)`,
            `  ${row.covers}`,
            // The number that turns a rate into a cost. Without it an agent
            // estimating a job would quietly assume one response per captcha.
            `  Typically ${row.typical_responses_per_captcha} response(s) per captcha`,
          );
        }
        lines.push(
          '',
          `Free (never reach the model): ${pricing.free_challenges.join(', ')}`,
          // The ceiling an agent should actually plan against: one captcha can
          // never cost more than this, however badly it goes.
          `One captcha costs at most ${pricing.max_billable_responses_per_session} billable responses ` +
            `($${((pricing.classes.find((row) => row.puzzle_class === 'image')?.usd_per_1000_responses ?? 0) * pricing.max_billable_responses_per_session).toFixed(2)} per 1,000 for images). ` +
            `Responses past that are served free, and the attempt is abandoned after ${pricing.max_responses_per_session}.`,
          `Packs: ${pricing.credit_packs.map((pack) => `$${pack.usd}`).join(', ')}`,
        );
        return text(lines.join('\n'));
      } catch (error) {
        return describe(error);
      }
    },
  );

  server.registerTool(
    'get_models',
    {
      title: 'Models',
      description:
        'The model lineup: which weights can be self-hosted, and which one the hosted endpoint ' +
        'runs. Useful for deciding whether to pay per solve or run it yourself.',
      annotations: { title: 'Models', readOnlyHint: true, openWorldHint: true },
    },
    async () => {
      try {
        const listing = await api.request<{
          base_url: string;
          models: Array<{
            name: string;
            zone: string;
            tagline: string;
            hosted: boolean;
            hugging_face_id: string | null;
            published: boolean;
            coming_soon: boolean;
            base_model: string;
            min_vram: string | null;
            weights_gb: number | null;
            accuracy: number | null;
            video: boolean;
          }>;
        }>('/api/v1/models', { authenticated: false });

        const lines = [`Hosted endpoint: ${listing.base_url}`, ''];
        for (const model of listing.models) {
          lines.push(`${model.name} — ${model.zone}`);
          lines.push(`  ${model.tagline}`);
          lines.push(`  Base: ${model.base_model}`);
          /*
           * `hosted` and "downloadable" are SEPARATE facts, and deriving one
           * from the other is what made this listing wrong in both directions
           * at once: it announced the unreleased model as "what the hosted API
           * runs" and reported the model actually taking requests as "reserved,
           * not uploaded yet". Twilight is downloadable AND serving; Abyss is
           * neither. Read each flag for what it says.
           */
          if (model.coming_soon) {
            lines.push('  Coming soon — not serving, nothing to download.');
          } else if (model.published) {
            lines.push(
              `  Weights: ${model.hugging_face_id} (~${model.weights_gb} GB, needs ${model.min_vram})`,
            );
          } else {
            lines.push(`  Weights: ${model.hugging_face_id} — reserved, not uploaded yet`);
          }
          if (model.hosted) lines.push('  This is what the hosted API answers with.');
          if (model.accuracy !== null) {
            lines.push(`  Measured: ${(model.accuracy * 100).toFixed(1)}% exact match`);
          }
          if (model.video) lines.push('  Handles video challenges.');
          lines.push('');
        }
        return text(lines.join('\n').trimEnd());
      } catch (error) {
        return describe(error);
      }
    },
  );

  return server;
}

type PollOutcome =
  | { kind: 'granted'; grant: TokenGrant }
  | { kind: 'waiting' }
  | { kind: 'denied' }
  | { kind: 'dead' };

/**
 * Poll until the token arrives or the budget runs out.
 *
 * `slow_down` doubles the interval, permanently for this attempt, which is what
 * RFC 8628 §3.5 asks for. It should not happen — we honour the interval the
 * server gave us — but a clock that jumps or a retried call can produce it, and
 * an implementation that ignores it is one that hammers an unauthenticated
 * endpoint whenever it does.
 */
async function pollForToken(api: ControlPlane, pending: PendingDevice): Promise<PollOutcome> {
  const deadline = Date.now() + SIGN_IN_WAIT_MS;
  let interval = Math.max(1, pending.intervalSeconds) * 1000;

  while (Date.now() < deadline) {
    if (pending.expiresAtMs <= Date.now()) return { kind: 'dead' };

    try {
      const grant = await api.request<TokenGrant>('/api/v1/device/token', {
        method: 'POST',
        authenticated: false,
        body: { device_code: pending.deviceCode },
      });
      return { kind: 'granted', grant };
    } catch (error) {
      if (!(error instanceof ApiError)) throw error;

      switch (error.code) {
        case 'authorization_pending':
          break;
        case 'slow_down':
          interval *= 2;
          break;
        case 'access_denied':
          return { kind: 'denied' };
        case 'expired_token':
          return { kind: 'dead' };
        default:
          throw error;
      }
    }

    // Do not overshoot the deadline sleeping: returning a moment early with
    // "call me again" is better than being killed by the client's timeout.
    const remaining = deadline - Date.now();
    if (remaining <= 0) break;
    await sleep(Math.min(interval, remaining));
  }

  return { kind: 'waiting' };
}
