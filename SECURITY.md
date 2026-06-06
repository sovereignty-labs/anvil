# Security Policy

## Supported Versions

Anvil is early-stage open source software. Security fixes are applied to the current `main` branch and the latest tagged release when releases are available.

| Version        | Supported   |
| -------------- | ----------- |
| latest release | ✅           |
| `main`         | ✅           |
| older releases | best effort |

## Reporting a Vulnerability

Please do **not** open a public GitHub issue for suspected security vulnerabilities.

Use GitHub private vulnerability reporting if available on this repository, or contact the maintainer privately through the project’s GitHub organization.

When reporting, please include:

* A clear description of the issue
* Steps to reproduce
* Affected version or commit
* Operating system and hardware details if relevant
* Whether the issue affects local execution, model/runtime management, MCP tooling, networking, or install/update behavior
* Any logs, command output, or proof-of-concept details that help confirm the issue

## Response Expectations

This project is maintained by a solo builder, so response times may vary, but security reports will be treated seriously.

Expected handling:

* Initial acknowledgement: best effort within 72 hours
* Triage and reproduction: as soon as practical
* Fix or mitigation plan: based on severity and impact
* Public disclosure: coordinated after a fix or mitigation is available when appropriate

## Scope

Security issues of interest include, but are not limited to:

* Unsafe install or update behavior
* Command injection or unsafe shell execution
* Path traversal or unsafe model file handling
* Exposure of local files, tokens, credentials, or environment variables
* Unsafe MCP tool behavior
* Unsafe network exposure of local inference services
* Privilege escalation or unintended access to remote/fleet nodes

## Out of Scope

The following are generally out of scope unless they demonstrate a concrete exploit path in Anvil itself:

* Vulnerabilities in upstream projects such as `llama.cpp`, operating systems, GPU drivers, or model files themselves
* General model behavior issues such as hallucination, prompt injection against a model, or unsafe model outputs
* Denial-of-service scenarios requiring full local control of the machine
* Reports based only on automated scanner output without a reproducible issue

## Disclosure

Please give the maintainer reasonable time to investigate and address valid reports before public disclosure.

Anvil’s goal is to remain transparent, local-first, and user-controlled. Responsible security reports help keep it that way.
