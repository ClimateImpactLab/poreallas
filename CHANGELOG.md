# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- Progress bar for forecast QDM script. ([@brews](https://github.com/brews), [PR#54](https://github.com/ClimateImpactLab/poreallas/pull/54))

- Testing for tas monthly histogram region extraction. ([@brews](https://github.com/brews), [PR#48](https://github.com/ClimateImpactLab/poreallas/pull/48))

### Changed

- Rename pre-processing scripts for consistency, clarity. ([@brews](https://github.com/brews), [PR#53](https://github.com/ClimateImpactLab/poreallas/pull/53))

- General documentation improvement, updates ([@brews](https://github.com/brews), [PR#36](https://github.com/ClimateImpactLab/poreallas/pull/36))

- Minor improvements to script readability. ([@brews](https://github.com/brews), [7d008e9](https://github.com/ClimateImpactLab/poreallas/commit/7d008e93efe9a044a57aba2ac52fb3f2a29630de))

### Removed

- Remove cruft FuzzyGridWeightingExtractor. ([@brews](https://github.com/brews), [PR#44](https://github.com/ClimateImpactLab/poreallas/pull/44))

## [0.5.2] - 2026-08-27

### Fixed

- Use sampled gamma in projected effects script to align with preliminary analysis. ([@brews](https://github.com/brews), [PR#40](https://github.com/ClimateImpactLab/poreallas/pull/40))

## [0.5.1] - 2026-08-25

### Fixed

- Script parameters, documentation used May projection settings, updated to August projection settings. ([@brews](https://github.com/brews), [PR#32](https://github.com/ClimateImpactLab/poreallas/pull/32))

- Use 100 quantiles in QDM by default, to align with preliminary analysis. ([@brews](https://github.com/brews), [PR#34](https://github.com/ClimateImpactLab/poreallas/pull/34))

- Forecast bias adjustment failing when forecast extends into new year. ([@brews](https://github.com/brews), [PR#31](https://github.com/ClimateImpactLab/poreallas/pull/31))

- GMFD extreme value cleaning moved so applies to all subsequent steps, not just ERA5 bias adjustment. ([@brews](https://github.com/brews), [PR#28](https://github.com/ClimateImpactLab/poreallas/pull/28))

## [0.5.0] - 2026-08-24

### Changed

- Update scripts and analysis for preliminary August forecast projection. ([@ezuetell](https://github.com/ezuetell), [PR#23](https://github.com/ClimateImpactLab/poreallas/pull/23))

## [0.4.0] - 2026-08-22

### Added

- Add preliminary analysis of May forecast and projection with QDM. ([@ezuetell](https://github.com/ezuetell), [PR#21](https://github.com/ClimateImpactLab/poreallas/pull/21))

### Changed

- BREAKING: Rewrite forecast download, climate data parsing, adding QDM bias correction and "noleap" calendar. ([@brews](https://github.com/brews), [PR#17](https://github.com/ClimateImpactLab/poreallas/pull/17))

## [0.3.0] - 2026-08-20

### Added

- Add preliminary analysis. ([@ezuetell](https://github.com/ezuetell), [PR#18](https://github.com/ClimateImpactLab/poreallas/pull/18))

## [0.2.0] - 2026-07-27

### Added

- Add prototype scripts for GMFD bias adjustment to ERA5 baseline and forecast ensemble projects. ([@kemccusker](https://github.com/kemccusker), [PR#13](https://github.com/ClimateImpactLab/poreallas/pull/13))

## [0.1.0] - 2026-07-24

- Initial running prototype.

[Unreleased]: https://github.com/climateimpactlab/poreallas/compare/v0.5.2...HEAD
[0.5.2]: https://github.com/climateimpactlab/poreallas/compare/v0.5.1...v0.5.2
[0.5.1]: https://github.com/climateimpactlab/poreallas/compare/v0.5.0...v0.5.1
[0.5.0]: https://github.com/climateimpactlab/poreallas/compare/v0.4.0...v0.5.0
[0.4.0]: https://github.com/climateimpactlab/poreallas/compare/v0.3.0...v0.4.0
[0.3.0]: https://github.com/climateimpactlab/poreallas/compare/v0.2.0...v0.3.0
[0.2.0]: https://github.com/climateimpactlab/poreallas/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/climateimpactlab/poreallas/releases/tag/v0.1.0
