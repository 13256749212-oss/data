# Campus-Scale 5G NR Radio Propagation Dataset — Processing and Reproducibility Code

This repository provides the processing and reproducibility code accompanying the campus-scale 5G NR radio propagation dataset collected at the Chenggong Campus of Yunnan University, Kunming, China.

The dataset combines **12 vehicle-based 5G NR drive-test sessions**, **3D terrain and building geometry**, **27 physical base stations associated with 79 verified Physical Cell Identifiers (PCIs)**, and **Sionna RT radio-propagation products** in a common spatial reference. The released workflow supports measurement preprocessing, station-parameter calibration, terrain-following per-station radio-map generation, network-wide best-server map generation, calibration-only network residual modeling, cell-level provenance export, and trajectory-disjoint quantitative evaluation.

> The large measurement and radio-map data products are distributed through Mendeley Data and are not duplicated in this code repository.

## Repository structure

```text
.
├── assets/
│   ├── ground.ply
│   └── ynu_chenggong_campus-001.ply
├── config/
│   ├── base_station_pci_mapping.csv
│   ├── station_catalog_27stations.csv
│   ├── coordinate_alignment.json
│   ├── source_metadata.json
│   ├── reference_geometry.csv
│   ├── global_epre_calibration.json
│   └── network_residual_calibration.json
├── metadata/
│   ├── spatial_reference.json
│   ├── scene_metadata.json
│   ├── propagation_configuration.json
│   ├── software_environment.json
│   ├── evaluation_reference.json
│   └── dataset_provenance.md
├── workflows/
│   ├── preprocessing/
│   ├── parameter_calibration/
│   ├── radio_map/
│   └── evaluation/
├── tools/
├── scripts/windows/
├── run_pipeline.py
├── check_project_layout.py
├── requirements.txt
└── environment.yml
```

Generated products are written to `outputs/` when the workflows are run. The external dataset is expected under `data/` and is intentionally excluded from this software repository.

## Released data products

The associated Mendeley Data release contains the following data groups.

| Data group | Main contents | Intended use |
|---|---|---|
| Measurement Dataset | 12 raw Cellular-Pro CSV files, aligned measurements, processed PCI–RSRP tables, and trajectory-split information | Measurement analysis, calibration, and independent evaluation |
| Radio-map Dataset | Original per-station Sionna RT maps, measurement-filled per-station maps, raw joint best-server map, and calibration-adjusted joint map | Propagation analysis, radio-map completion, and dataset-informed coverage analysis |
| Calibration Data | Station-parameter calibration outputs, reference geometry, network power alignment, and calibration-only residual-model outputs | Reproduction of the calibrated propagation workflow |
| Evaluation | Calibration and held-out matched tables, overall metrics, stratified metrics, and residual-map data | Quantitative validation and error analysis |
| Scene and Reference Data | Terrain mesh, building mesh, physical base-station locations, PCI mappings, and sector information | Spatial alignment and Sionna RT scene reconstruction |
| Metadata | Spatial-reference, scene, propagation, software-environment, provenance, and evaluation metadata | Long-term reuse and reproducibility |

## Reproduction environment

The results reported in the accompanying data article were generated using:

- **Python 3.10**
- **Sionna RT 1.2.2**
- **Blender 4.5 LTS**

Create the environment with:

```bash
conda env create -f environment.yml
conda activate sionna_env
```

or install the Python dependencies and Sionna separately:

```bash
python -m pip install -r requirements.txt
python -m pip install sionna==1.2.2
```

A CUDA-capable NVIDIA GPU is recommended for ray-tracing calibration and radio-map generation.

## Required data placement

After downloading the dataset from Mendeley Data, place the measurement data under the repository root as follows:

```text
data/
├── raw_measurements/
├── aligned_measurements/        # optional if regenerated
└── processed/                   # optional if regenerated
```

The main long-format measurement table used by the calibration and evaluation workflows is:

```text
data/processed/cell_pci_rsrp_long_27stations.csv
```

The table retains the source measurement trajectory so that complete drive-test sessions can be separated into calibration and held-out evaluation subsets.

## Workflow

Run all commands from the repository root.

### 1. Check the repository

```bash
python run_pipeline.py --help
python run_pipeline.py check
```

The layout check verifies the required scene meshes, station/PCI configuration, coordinate metadata, and available measurement-data stage.

### 2. Prepare the drive-test measurements

```bash
python run_pipeline.py prepare-data
```

To overwrite previously generated intermediate files:

```bash
python run_pipeline.py prepare-data --force
```

The same preprocessing sequence can be run step by step:

```bash
python run_pipeline.py align
python run_pipeline.py extract
python run_pipeline.py preprocess
```

The spatial alignment is:

```text
WGS84 (EPSG:4326) -> EPSG:3857 -> Blender-local coordinates
```

Receiver elevation follows the local terrain at **DEM + 1.5 m**. The nominal 1 projected-m grid interval is defined in the EPSG:3857-derived local coordinate system and should not be interpreted as exactly 1 geodetic ground metre. For analyses requiring accurate metric ground distance, the original WGS84 coordinates can be reprojected to a local metric CRS such as EPSG:32648 (UTM zone 48N).

Main processed outputs include:

```text
data/aligned_measurements/
data/processed/extracted/
data/processed/cell_pci_rsrp_long_27stations.csv
data/processed/cell_pci_rsrp_1m_calibration.csv
data/processed/cell_pci_rsrp_2p77m_localization.csv
```

### 3. Calibrate the physical base stations

A quick single-station run can be used to verify the Sionna RT environment:

```bash
python run_pipeline.py calibrate --stations 3 --quick
```

Run the released calibration workflow for all 27 physical base stations:

```bash
python run_pipeline.py calibrate --stations all
```

The 12 complete drive-test sessions are divided by trajectory into **nine calibration trajectories** and **three held-out evaluation trajectories**. Station-parameter estimation, network-wide power alignment, and residual-model selection use the calibration trajectories only.

The released workflow uses `config/reference_geometry.csv` for the reference station geometry during routine regeneration. Geometry can be intentionally re-estimated with the dedicated command-line option when repeating the geometry-estimation stage.

Common propagation and calibration settings are:

- center frequency: **2.565 GHz**
- bandwidth: **100 MHz**
- resource blocks: **273**
- maximum propagation depth: **5**
- diffraction: **enabled**
- edge diffraction: **enabled**
- transmitter array: generic **8 × 4** planar array with 3GPP TR 38.901-style element pattern
- receiver surface: **DEM + 1.5 m**
- carrier-power search: **50–55 dBm**
- network-wide EPRE alignment: **+3.5353613488734865 dB**
- power-regularization coefficient: **0.25**

The same propagation physics are used for calibration, per-station radio maps, and the network-wide joint map.

Main calibration outputs:

```text
outputs/parameter_calibration/
├── all_27stations_summary.csv
├── estimated_initial_directions_27stations.csv
├── calibration_validation_split.csv
├── calibration_validation_split.json
└── station_*/
```

### 4. Generate the terrain-following per-station radio maps

```bash
python run_pipeline.py export-dem --stations all
```

Each physical station is evaluated on a nominal **512 × 512 projected-m** window with a nominal **1 projected-m** horizontal interval. The receiver surface follows the DEM at +1.5 m.

Two per-station products are retained:

- **Original Sionna RT maps:** finite simulation results only.
- **Measurement-filled maps:** same-station measurements are inserted only where the corresponding original Sionna RT value is missing; valid simulated values are never overwritten.

Cell-level provenance and measurement-availability masks are stored with the numerical products.

Main output root:

```text
outputs/bestparam_radio_maps_512m/
```

### 5. Generate the network-wide raw joint best-server map

Optional checks:

```bash
python run_pipeline.py export-joint-map --dry-run
python run_pipeline.py export-joint-map --quick
```

Generate the complete map:

```bash
python run_pipeline.py export-joint-map
```

The joint product covers **4000 × 3000 nominal projected metres** and is assembled from **48 tiles** of 500 × 500 projected metres. At each finite outdoor grid cell, the strongest candidate PCI determines the raw best-server RSRP, PCI, physical-station identifier, and sector index.

The released reference geometry contains:

- **11,692,901 outdoor receiver cells**
- **7,972,969 finite best-server cells**

Main output:

```text
outputs/joint_best_server_4000x3000/
└── joint_best_server_27stations_4000x3000.npz
```

### 6. Fit the calibration-only network residual model

```bash
python run_pipeline.py fit-network-residual
```

The raw Sionna RT joint map is retained unchanged. A separate calibration-adjusted joint RSRP map is generated using a low-capacity residual model selected only from the nine calibration trajectories by leave-one-trajectory-out cross-validation. The residual layer adjusts RSRP values only and does not reselect the best station or best PCI.

Main outputs:

```text
outputs/network_residual_calibration/
├── cv_candidates.csv
└── calibration_matched_cells.csv

outputs/joint_best_server_4000x3000/
└── joint_best_server_27stations_4000x3000_calibrated.npz
```

### 7. Export cell-level provenance

```bash
python run_pipeline.py export-provenance
```

The provenance codebook is:

```text
0 = missing / invalid
1 = original finite Sionna RT
2 = measurement-filled same-grid value
3 = Sionna RT plus calibration-only network residual
```

Measurement availability and calibration/evaluation observation masks are stored separately. This distinction prevents a valid simulation cell that is spatially coincident with a field measurement from being mislabeled as measurement-derived.

Main outputs:

```text
outputs/joint_best_server_4000x3000/
├── joint_best_server_27stations_4000x3000_provenance.npz
├── joint_map_provenance_summary.json
└── joint_map_provenance_counts.csv
```

### 8. Evaluate the raw and calibration-adjusted joint maps

```bash
python run_pipeline.py compare-joint-map
python run_pipeline.py compare-calibrated-map
python run_pipeline.py export-evaluation
```

Measurements are mapped to the same nominal projected grid as the joint map. Repeated best-server observations falling in the same grid cell are median aggregated within each trajectory subset.

The released split contains:

- **8,031 calibration matched grid cells**
- **2,380 held-out evaluation matched grid cells**

Across the 2,380 held-out cells, the raw joint map has an RMSE of **16.20 dB**. The calibration-adjusted joint map yields:

- RMSE: **12.06 dB**
- MAE: **9.89 dB**
- median absolute error: **8.77 dB**
- mean simulation-minus-measurement bias: **−0.34 dB**
- error standard deviation: **12.05 dB**
- Pearson correlation: **0.227**
- P90 absolute error: **19.26 dB**
- P95 absolute error: **22.67 dB**

The three held-out trajectories were collected during the same campaign and were available during method development. The reported result is therefore a **trajectory-disjoint internal evaluation**, not an independently acquired external test.

Evaluation outputs are written under:

```text
outputs/evaluation/
```

They include raw and calibration-adjusted comparison tables, per-station and per-PCI metrics, transmitter-distance and terrain-elevation stratification, residual-map data, and summary metrics.

### 9. Export reproducibility metadata

```bash
python run_pipeline.py export-metadata
```

The machine-readable metadata document:

- the WGS84 -> EPSG:3857 -> Blender-local coordinate transformation and local origin;
- terrain/building mesh hashes and spatial bounds;
- scene-material assumptions;
- antenna configuration;
- calibration search ranges;
- propagation solver settings;
- SS-RSRP mapping assumptions;
- software environment;
- joint-map reference statistics;
- cell-level provenance definitions; and
- evaluation reference metrics.

## SS-RSRP mapping

For the 100 MHz configuration:

```text
N_RB = 273
subcarriers_per_RB = 12
N_active = 273 × 12 = 3276
N_SSS_RE = 127
```

The configured **3276 active subcarriers** are used only to convert the station carrier-power parameter to a uniform per-active-subcarrier energy-per-resource-element (EPRE) approximation on the transmitter side. They are **not** the SS-RSRP averaging count.

Measurement-comparable SS-RSRP is represented as the linear average over the **127 SSS-bearing resource elements** under the narrowband path-gain approximation documented in `metadata/propagation_configuration.json`.

## Data-use guidance

Use the released products according to the intended task:

- **Original Sionna RT maps:** propagation-model analysis and evaluation using measurements not used for fitting.
- **Measurement-filled maps:** radio-map completion/reconstruction studies. Measurement-filled cells must not be treated as independent simulation ground truth.
- **Calibration-adjusted joint map:** dataset-informed RSRP surface for the campus; use together with the provenance metadata.
- **Raw and processed measurements:** measurement-driven analyses and independent reprocessing.
- **Held-out evaluation tables:** quantitative assessment of the released raw and calibration-adjusted joint maps.

## Limitations

The dataset represents one mountainous university campus, one 5G SA/n41 configuration, one measurement-device setup, and measurements restricted to vehicle-accessible roads. The scene contains terrain and principal buildings but does not explicitly model vegetation, parked or moving vehicles, pedestrians, small street furniture, or detailed facade and roof structures.

The transmitter model uses a generic Sionna/3GPP TR 38.901-style 8 × 4 array rather than the vendor-specific commercial antenna pattern. The true base-station height, effective transmit power, azimuth, and downtilt were unavailable and were estimated from calibration measurements within bounded search ranges.

Because EPSG:3857 is retained for compatibility with the distributed scene geometry, nominal projected-grid intervals should not be interpreted as exact ground distances. Analyses requiring accurate physical distance should use an appropriate local metric CRS.

The measurement campaign does not characterize seasonal vegetation changes, weather variability, day-to-day network optimization, traffic conditions, other terminal hardware/firmware, other receiver heights, other carriers, or other frequency bands. The dataset is therefore intended primarily as a reproducible methodological benchmark for this campus environment rather than as evidence of direct generalization to all 5G propagation environments.

## Citation and data access

When using the dataset, please cite the Mendeley Data record associated with the data release and the accompanying Data in Brief article when available.

## License

Please follow the license terms stated in the Mendeley Data record and the Zenodo software archive.
