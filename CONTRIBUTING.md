# Contributing

Issues and pull requests are welcome.

This crate is meant to stay small and inspectable. The best contributions make
audio preparation clearer, safer, or easier to run.

## Good first areas

- Add small real-audio fixtures with clear license provenance.
- Improve README examples for converter, chunker, and preprocessing usage.
- Add benchmarks for VAD, conversion, and chunking.
- Expand format decoding beyond WAV without overstating support.
- Improve error messages for malformed audio inputs.

## Development

```bash
cargo test
cargo test --features fixtures
cargo clippy -- -D warnings
cargo fmt --check
```

## AI-assisted work

I use AI tools for drafting, review, refactoring ideas, and test ideation. All
code and documentation in this repository is reviewed, edited, and tested by me
before release.

Contributions that use AI tools are welcome if you can explain the change, keep
the diff focused, run the relevant tests, and disclose substantial AI assistance
in the pull request description.

## Ground rules

- Keep examples runnable without private files.
- Keep format-support claims aligned with code.
- Avoid adding large binary fixtures without a clear reason and license.

## License

Contributions are accepted under the same MIT OR Apache-2.0 license as the
project.
