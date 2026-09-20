# Kitty Bridge — project instructions

Project-level instructions for Claude Code sessions working in this
repository. They add to the user-level instructions; where the two overlap,
both apply.

## Resolving GitHub code reviews

When you take on a GitHub code-review conversation on this repository — a
review round, a comment thread, an inline review thread — the work is not
done when the code is fixed. Do all of the following:

1. **Enumerate before you answer.** List every open conversation and every
   review comment on the pull request, including inline review threads,
   which a plain comments listing does not return. Assign each one a
   disposition (addressed, deferred, declined) before responding to any.
2. **Post a PR-level comment summarising the round.** State what was
   addressed and implemented, and what was deferred (with the reason).
   The summary is one comment, posted once per round, on the pull request
   itself.
3. **Reply in thread where detail helps.** A conversation whose fix needs
   explanation — a design choice, a follow-up ticket, a rejected
   alternative — gets that explanation as a reply in its own thread, not
   only in the summary comment.
4. **Resolve what you fixed.** Resolve every conversation that was taken
   and fixed. A conversation you will not address is never silently
   resolved and never resolved without a stated reason: post an
   in-thread reply naming the reason and leave the conversation open.

These rules apply to the agent working the review; the CI review bot
comments only and resolves nothing, and so is out of scope today.
