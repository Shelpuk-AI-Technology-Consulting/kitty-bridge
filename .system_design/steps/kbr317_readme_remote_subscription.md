---
id: kbr317_readme_remote_subscription
depends_on: [kbr317_cli_device_flag]
---

# README: remote / headless ChatGPT subscription setup + GPT-6 Astra

KBR-317, step 5 of 6. The setup documentation the ticket explicitly asks for
("indicate clearly how I set up OpenAI Astra as a model with Kitty Bridge and
an OpenAI subscription").

## Why

The user's exact scenario — Claude Code on a remote Ubuntu CLI-only VM, ChatGPT
subscription, browser on a laptop — needs a recipe, not just a flag. The README
also needs to name the Astra model ids because the Codex subscription backend's
model list rotates independently of kitty releases; the docs list the current
candidates and tell the reader to use whichever their subscription advertises.

## To Be

A new README section "ChatGPT subscription on a remote / headless machine":

1. Why the original flow fails (browser hits the wrong localhost).
2. The default behaviour on an SSH session: the device-code prompt, with a
   full transcript.
3. The fallback: `ssh -L 1455:localhost:1455 user@vm` + `--no-device-auth`,
   **plus a note that port 1455 must be free on the VM** (pre-existing
   kitty limitation: the redirect URI is hardcoded to 1455 even though
   the callback server scans 1455-1459; not fixed in this change).
4. A "Use GPT-6 Astra" subsection: run `kitty auth openai`, pick the Astra
   model id at the model prompt (candidates: `gpt-6-astra-codex`,
   `gpt-6-astra`; check what the ChatGPT UI / codex offers on the
   subscription), then launch with `kitty claude`.

Guard: no literal `KBR-` references in the README (the PyPI long-description
guard, kbr277 precedent).

## Acceptance

- The section, followed verbatim, lets a personal ChatGPT Plus/Pro
  subscription holder on a remote Ubuntu VM authenticate kitty and select
  Astra as the model.
- `grep -E "KBR-[0-9]+" README.md` returns nothing.
- The existing README table guards / count pins are not disturbed.

## Implementation notes

- New `### ChatGPT subscription on a remote or headless machine` section
  after the existing ChatGPT quick-start example: failure explanation,
  device-flow transcript, admin-toggle note, `--no-device-auth` tunnel
  fallback with the port-1455 caveat, and `#### Using GPT-6 Astra` with
  `gpt-6-astra-codex` / `gpt-6-astra` / `gpt-6-astra-pro` candidates and
  the "check what your plan advertises" guidance.
- Commands table gained the `kitty auth openai` rows (the command existed
  in the router but was undocumented).
- `grep -cE "KBR-[0-9]+" README.md` → 0 matches (guard holds).
