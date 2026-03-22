# Security Policy

## Supported Versions

| Version | Supported |
|---------|-----------|
| 2.x     | ✅ Yes     |
| < 2.0   | ❌ No      |

## Reporting a Vulnerability

Please **do not** open a public GitHub issue for security vulnerabilities.

Use one of the following channels:

- **Private vulnerability report (preferred):** Use the
  [Report a vulnerability](https://github.com/The-Schultz-Lab/plottle/security/advisories/new)
  button on the Security tab. This keeps the report private until a fix is available.
- **Email:** Contact the maintainer directly at the address listed on the
  [NCCU Schultz Lab GitHub profile](https://github.com/The-Schultz-Lab).

## What to Include

A useful report includes:

- A description of the vulnerability and its potential impact
- Steps to reproduce (minimal example if possible)
- The version of Plottle affected
- Any suggested fix or mitigation

## Response Timeline

- **Acknowledgement:** within 5 business days
- **Assessment and triage:** within 2 weeks
- **Fix or workaround:** depends on severity; critical issues are prioritised

## Known Limitations by Design

The following behaviors are **intentional** and documented — please do not report
them as vulnerabilities unless you have found a bypass of the existing mitigations:

- **`eval()` in Analysis Tools and Data Tools:** User-provided function expressions
  (e.g., custom curve-fit formulas) are evaluated with `eval()`. The execution context
  is restricted to a minimal namespace. This is acceptable for a local/classroom tool
  but should not be exposed to untrusted users on a public server without additional
  sandboxing.

## Scope

This policy covers the `plottle` package and its Streamlit GUI and CLI. It does not
cover third-party dependencies — please report those to their respective maintainers.
