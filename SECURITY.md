# Security Policy

`speech-prep` is a local audio preprocessing crate, not a hosted service. The
main risks are malformed audio inputs, privacy mistakes around user audio, and
incorrect documentation about supported formats.

## Supported Versions

Security fixes target the current `main` branch and the latest crates.io release.

## Reporting A Security Issue

Do not attach private audio, private transcripts, API keys, or proprietary
datasets to public issues.

For security-sensitive reports, use GitHub private vulnerability reporting if it
is available for this repository. If it is not available, open a public issue
with only a short non-sensitive summary and ask for a private contact path.

For normal decoding bugs, crashes, and format-support gaps, open a regular
GitHub issue with the command you ran, the Rust version, the file format, sample
rate, channel count, and whether the audio is public or synthetic.
