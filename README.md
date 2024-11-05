# Introduction

This project is purely a developer tool. If you are a ZLUDA user, it's not useful to you.

This project tests CUDA PTX compiler API. The main goal of this project is to ensure that ZLUDA's PTX compiler is compatible with NVIDIA PTX compiler.

Each test case in the suite tests a particular instruction by either:
* testsing all possible inputs if the input size is small anough (4 bytes or less)
* testing against a large number of randomly generated inputs

## Limitations
* This project requires nightly Rust compiler. This is due to limitations in Rust's f16 and floating point rounding support.
* By default this projects builds with `target-cpu=native`. Running CPU-side verificaiton takes a lot of time and this improves the run times.

# Usage

## List tests

Print all possible test cases to stdout:
```
cargo +nightly run -r -- -l
```

## Run tests

Run tests using CUDA library at `<CUDA_LIB>` path and matching `<REGEX_FILTER>` regex:

```
cargo +nightly run -r -- <CUDA_LIB> -f <REGEX_FILTER>
```


## Help

Print help message:

```
cargo +nightly run -r -- -h
```

# License

This software is dual-licensed under either the Apache 2.0 license or the MIT license. See [LICENSE-APACHE](LICENSE-APACHE) or [LICENSE-MIT](LICENSE-MIT) for details