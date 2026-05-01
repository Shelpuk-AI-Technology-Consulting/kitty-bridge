from __future__ import annotations


def is_cloudflare_block(status: int, body: str) -> bool:
    if status != 403:
        return False
    return get_cloudflare_signature(body) is not None


# Order matters: specific signatures first, generic "cloudflare" last.
# get_cloudflare_signature() returns the first match, so the generic
# "cloudflare" catch-all only fires when no specific signature is present.
_CF_SIGNATURES = ("cf-mitigated", "_cf_chl_opt", "cf-browser-verification", "cloudflare")


def get_cloudflare_signature(body: str) -> str | None:
    """Return the first Cloudflare body signature found, or None."""
    lower = body.lower()
    for sig in _CF_SIGNATURES:
        if sig in lower:
            return sig
    return None
