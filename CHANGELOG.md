# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Fixed

- GMFD extreme value cleaning moved so applies to all subsequent steps, not just ERA5 bias adjustment. ([@brews](https://github.com/brews), [PR#17](https://github.com/ClimateImpactLab/poreallas/pull/28))

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

[Unreleased]: https://github.com/climateimpactlab/poreallas/compare/v0.5.0...HEAD
[0.5.0]: https://github.com/climateimpactlab/poreallas/compare/v0.4.0...v0.5.0
[0.4.0]: https://github.com/climateimpactlab/poreallas/compare/v0.3.0...v0.4.0
[0.3.0]: https://github.com/climateimpactlab/poreallas/compare/v0.2.0...v0.3.0
[0.2.0]: https://github.com/climateimpactlab/poreallas/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/climateimpactlab/poreallas/releases/tag/v0.1.0
