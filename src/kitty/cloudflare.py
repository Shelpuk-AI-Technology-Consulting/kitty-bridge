from __future__ import annotations


def is_cloudflare_block(status: int, body: str) -> bool:
    if status != 403:
        return False
    return get_cloudflare_signature(body) is not None


# F48: Only specific Cloudflare signatures are matched.  The bare word
# "cloudflare" was removed because it false-positives on any 403 body that
# mentions Cloudflare in a URL, support contact, or incident report.
_CF_SIGNATURES = ("cf-mitigated", "_cf_chl_opt", "cf-browser-verification")


def get_cloudflare_signature(body: str) -> str | None:
    """Return the first Cloudflare body signature found, or None."""
    lower = body.lower()
    for sig in _CF_SIGNATURES:
        if sig in lower:
            return sig
    return None
