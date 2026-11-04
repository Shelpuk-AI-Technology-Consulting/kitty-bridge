---
id: brotli_subscription_relay
depends_on: [kbr319_exact_messages_relay]
---

# Brotli decoding for the Anthropic subscription relay

## Planned delivery

Declare `Brotli>=1.2` as a direct runtime dependency. This lets aiohttp decode an Anthropic
response whose `Content-Encoding` is `br`, while the relay continues to pass Claude Code's
end-to-end request headers and upstream response bytes unchanged at its public boundary.

## Decisions and why

- **Direct `Brotli` dependency, not `aiohttp[speedups]`.** aiohttp documents that its `br`
decoder requires Brotli. The `speedups` extra would also add `aiodns`, changing DNS resolution
for every provider route merely to obtain a response decoder. A direct dependency is the smaller,
explicit contract.
- **`Brotli>=1.2`, because that is aiohttp's own requirement.** Since aiohttp 3.13.3 aiohttp needs
  Brotli 1.2 or newer, and 3.13.3 sits inside the `aiohttp>=3.11,<3.14` range this package
  supports. Kitty does not install the `speedups` extra that would impose the floor, so it is
  stated here.
- **Do not filter `Accept-Encoding`.** Removing `br` would avoid the decoder need but violates
the byte-exact subscription-relay contract: `Accept-Encoding` is an end-to-end client header and
must reach Anthropic unchanged.
- **Do not add a zstd dependency for now.** A probe on 2026-11-04, which only ever saw 429 error
replies, showed `br` when offered and no compression when `zstd` was offered alone or with gzip and
deflate. That does not prove Anthropic never sends zstd; SYSTEM_DESIGN §16.1 names the log line that
reverses this.

## Implementation notes

Landed. `pyproject.toml` declares `Brotli>=1.2` directly, with a comment recording why the
decoder is a product dependency and why `aiohttp[speedups]` was rejected as the carrier.

Two loopback tests (collected as `l1`) in `tests/bridge/test_subscription_messages_relay.py` drive the relay against a
local upstream serving brotli-compressed bytes — one for the plain arm, one for the streaming
arm. The compressed form is committed as bytes, so the test needs no encoder at run time. The
upstream is a real loopback server rather than `aioresponses`, because `aioresponses` returns
the body verbatim and never runs the client-side decoder: against it the test asserted only that
the compressed bytes were forwarded, which no decoder would change. With the real server the
test reproduces the production failure exactly — `ClientResponseError: 400, message='Can not
decode content-encoding: brotli (br). Please install `Brotli`' — when the decoder is absent.

One L2 packaging test in `tests/test_pypi_packaging.py` pins the `Brotli` requirement, so
removing the declaration fails the suite rather than a fresh install.

Verified live on 2026-11-04: a two-member pool (`anthropic` primary, `meta` backup-tier) served
a Claude Code request through the relay, and the session summary recorded the `anthropic`
member with one attempt and one completion while `meta` had none. Before the fix the same
scenario died in the decoder. Repeated with `anthropic` positioned last in the member list, as
it is in the operator's `my` pool, and with a request carrying a 269 KB file — two requests,
two completions, zero failovers, all served by the `anthropic` member.

Each new assertion was watched failing against a deliberately broken build, which
is the cheap form of mutation validation for a test that has no product code to
mutate. Adding `accept-encoding` to `_SUBSCRIPTION_STRIPPED_HEADERS` in
`src/kitty/bridge/server.py` made both forwarding assertions fail with a
message that names the consequence; hiding the Brotli decoder made both decode
tests fail with the production error. Both defects were reverted immediately
afterwards and `src/kitty/` is byte-identical to `HEAD`.

The forwarding premise is pinned separately and at the cheaper layer:
`test_relay_headers_strips_full_hop_set` asserts that `Accept-Encoding` survives the relay.
That is the header the upstream reads to choose its response encoding, so it is the premise the
decoder requirement rests on; without that assertion, stripping the header would silently
retire the Brotli dependency while the decode tests kept passing.

The two decode tests bind a second real socket (the upstream), so the module is socket-binding
by the §8.2 definition. It is **not** added to `SOCKET_BINDING_L1_MODULES`: that registry also
feeds the nightly mutation run's `--ignore` rows, so registering it would drop this module's
relay coverage from the mutation baseline — a change to a nightly job wider than this fix. The
module already bound a socket (the bridge itself) before this step and was already unregistered,
so this step widens an existing gap rather than opening one. The trade-off is recorded here so
the next maintainer sees it: the decode tests are deterministic (twelve consecutive runs, and
the whole Fast gate), and the nightly mutation run's exposure is unchanged from what it already
carried for this module.

This step is a dependency and its guards only; no relay routing, health, or failover behaviour
moved. It was necessary but not sufficient: the member still stalled mixed sessions until
`subscription_relay_recovery`.
