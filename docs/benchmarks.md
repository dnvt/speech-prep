# Benchmarks

No formal benchmark numbers are published for `speech-prep` yet.

The CI checks correctness, docs, examples, and packaging. It does not currently
publish latency or throughput results.

## How To Add A Benchmark Result

When adding a number, include:

- command;
- commit hash;
- Rust version;
- hardware and operating system;
- input format, duration, sample rate, and channel count;
- whether the input is synthetic, public, or private;
- warmup count and number of measured runs.

## What Not To Claim

Do not turn example output or a one-off local run into a general performance
claim. Audio timing changes with format, sample rate, channel count, CPU, and
feature flags.
