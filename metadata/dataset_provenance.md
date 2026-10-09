# Dataset provenance and recommended use

## Spatial reference
The released scene and measurements use a WGS84 -> EPSG:3857 -> local Blender transform. EPSG:3857 is retained for internal consistency with the released scene. The nominal 1 m map interval is an internal projected/local-coordinate interval and is not an exact geodetic ground-distance interval. For analyses that require accurate metric ground distances, users should transform the original WGS84 coordinates to an appropriate local metric CRS such as WGS 84 / UTM zone 48N (EPSG:32648).

## Radio-map provenance codes
Per-PCI measurement-filled NPZ products include `value_source_mask`:

- `0`: missing/invalid (including masked building cells)
- `1`: original finite Sionna RT result
- `2`: Sionna RT was missing and the cell was filled by a same-PCI measurement from the same 1 m grid cell

`measurement_available_mask` is stored separately so a user can identify measured locations even when a valid Sionna RT value was retained.

## Recommended use
- Original Sionna RT maps: independent propagation-model validation and simulation-only studies.
- Measurement-filled maps: dense-reference/reconstruction studies; not independent ground truth when the same measurements were used during model fitting or training.
- Raw/processed measurements: measurement-driven analyses.
- Matched measurement-simulation records: direct model-measurement quality characterization.

## Calibration and validation
The calibration workflow creates a complete-source-file (trajectory-level) calibration/validation split. Base-station direction estimation and parameter fitting use calibration trajectories only. `compare-joint-map` uses the held-out validation trajectories by default.


## Terrain and building provenance
The terrain reference originates from Web-Mercator elevation data obtained through a QGIS DEM/elevation plugin. The native resolution of those downloaded source tiles was not separately recorded. The released DEM-following receiver surfaces are evaluated on the dataset's 1 m analysis grid at terrain elevation + 1.5 m. Campus buildings were manually modeled from Web-Mercator-projected satellite imagery, using the terrain/ground surface as the vertical reference; the model is simplified propagation geometry rather than a facade-detail reconstruction.

## SS-RSRP power mapping
The receive-side SS-RSRP quantity follows the TS 38.215 SSS-RE linear-average definition. The released baseline does not fit a second station-specific SS-RSRP/SSS-RE amplitude offset. Instead, the modeled station/sector carrier power (`shared_power_dbm`, searched only within 50-55 dBm) is mapped to SSS-RE EPRE by uniform allocation across the configured `273 * 12 = 3276` active subcarriers. The 3276 count is therefore a TX-side allocation assumption and is not the 3GPP SS-RSRP averaging count; the receive-side average remains over the 127 SSS REs. A single network-global EPRE alignment offset is estimated from calibration trajectories only, shared by all stations, and frozen before held-out validation. The released network-global EPRE alignment is +3.535361 dB and is stored in `config/global_epre_calibration.json`. Held-out evaluation results are recorded in `metadata/evaluation_reference.json`; no held-out evaluation metric is used for power alignment.

## Calibration-adjusted joint map and network provenance
The released raw joint-map calibration uses reliability-weighted leave-one-calibration-trajectory-out selection with the reference station geometry frozen. The selected network-global EPRE offset is `+3.5353613488734865 dB` with power-regularization lambda `0.25`.

The calibration-adjusted joint map applies a low-capacity network residual model selected only from the nine calibration trajectories. The selected model is `affine_rsrp_distance` with ridge `10.0`; the three held-out trajectories are explicitly excluded from fitting and model selection. The raw Sionna RT map remains available and is not overwritten.

The network-scale provenance companion uses the following source codebook:

- `0`: missing/invalid
- `1`: original Sionna RT
- `2`: measurement-filled same-grid value (reserved for per-PCI measurement-filled products)
- `3`: Sionna RT value after the calibration-only network residual layer

A separate `measurement_split_mask` marks whether a cell was visited by a calibration trajectory, a held-out trajectory, both, or neither. This observation mask does not imply that the field measurement replaced the map value.

The three held-out trajectories were collected in the same measurement campaign. The reported result is therefore described as a trajectory-disjoint internal evaluation rather than as an independently acquired external test.
