# Contributing

Issues and pull requests are welcome.

## Good areas to contribute

- New audio format support
- Additional preprocessing filters
- VAD algorithm improvements
- Performance benchmarks on different hardware
- Documentation improvements

## Development

This crate currently targets Rust 1.87.

Run the full verification set before opening a pull request:

```bash
cargo test
cargo test -- --ignored
cargo clippy -- -D warnings
cargo fmt --check
cargo doc --no-deps
cargo run --example vad_detect
cargo package --allow-dirty
```

Optional checks:

```bash
cargo test --features fixtures
```

Call out public API changes in the pull request description and update `CHANGELOG.md` when behavior or compatibility changes.

## License

Contributions are accepted under the same MIT OR Apache-2.0 license as the project.
