# Security Policy

## Supported versions

Only the latest release gets security fixes. If you can, check that the issue still happens on it before reporting.

## Reporting a vulnerability

Please do not open a public issue. Report it privately through [GitHub private vulnerability reporting](https://github.com/mhajder/webex-bot-ai/security/advisories/new).

Include the affected version, how to reproduce it and what an attacker could do with it. Do not send real tokens or API keys; redact them.

I aim to acknowledge a report within 7 days and will keep you updated until it is fixed and released.

## Scope

In scope is this bot, for example:

- the Webex token, LLM API keys or MCP auth tokens leaking into logs, Webex messages or Sentry events
- the bot answering users, domains or rooms outside `WEBEX_APPROVED_USERS`, `WEBEX_APPROVED_DOMAINS` or `WEBEX_APPROVED_ROOMS`
- conversation history or summaries from one space showing up in another
- an MCP auth token being sent to a different server than the one it is configured for

Out of scope are wrong or unsafe answers from the model itself, and vulnerabilities in Webex, LiteLLM, the LLM providers or the MCP servers you connect to. Report those to the respective project.
