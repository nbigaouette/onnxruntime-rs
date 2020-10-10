# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- Integrate Runtime's logging with rust's `tracing` logging (#21)
- Update `ureq` used to download pre-trained models, fixing download problem (algesten/ureq#179).
- Properly support inputs with dynamic dimensions (#23)

## [0.0.8] - 2020-08-26

### Added

- Use ONNX Runtime 1.4.0

## [0.0.7] - 2020-08-26

### Added

- Use `tracing` crate instead of `log` (#19)
- Add integration tests (#17)
- Add possibility to download most pre-trained models available from [ONNX Zoo](https://github.com/onnx/models) (#16)

## [0.0.6] - 2020-08-14

### Added

- Add feature to download pre-trained pre-trained models available from [ONNX Zoo](https://github.com/onnx/models) (#15)
- Add coded coverage measurement (#13)
- API renames and cleanups

## [0.0.5] - 2020-08-9

### Added

- Initial working version

[Unreleased]: https://github.com/nbigaouette/onnxruntime-rs/compare/v0.0.8...HEADD
[0.0.8]: https://github.com/nbigaouette/onnxruntime-rs/compare/v0.0.7...v0.0.8
[0.0.7]: https://github.com/nbigaouette/onnxruntime-rs/compare/v0.0.6...v0.0.7
[0.0.6]: https://github.com/nbigaouette/onnxruntime-rs/compare/v0.0.5...v0.0.6
