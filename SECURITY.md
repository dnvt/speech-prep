# Security Policy

`speech-prep` parses untrusted audio files, so malformed input that causes
a panic, a hang, or excessive memory use is treated as a security issue.

## Supported versions

Fixes go into the latest release on crates.io and `main`.

## Reporting

Please use GitHub's
[private vulnerability reporting](https://github.com/dnvt/speech-prep/security/advisories/new)
rather than a public issue. Include a minimal input file that reproduces
the problem if you can. Don't send recordings of real people; a generated
or truncated file is almost always enough.

For ordinary bugs, open a regular issue with the Rust version, the input
format, sample rate, and channel count.
