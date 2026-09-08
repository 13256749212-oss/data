# Campus-Scale 5G NR Radio Propagation Dataset

This repository provides the data-processing, ray-tracing, evaluation, localization, and radio-map reconstruction code associated with the campus-scale 5G NR radio propagation dataset collected at the Chenggong Campus of Yunnan University, Kunming, China.

The dataset integrates **12 vehicle-based 5G NR drive-test sessions**, **3D terrain and building geometry**, **27 physical base stations associated with 79 PCIs**, and **Sionna RT radio maps** within a common spatial reference. The released workflows support measurement preprocessing, base-station parameter calibration, per-station and network-scale radio-map generation, measurement–simulation comparison, physical base-station localization, and sparse radio-map reconstruction.

**Dataset DOI:** https://doi.org/10.17632/d5xrrgsj4f.1  
**Code repository:** https://github.com/13256749212-oss/data

<p align="center">
  <img src="docs/images/figure1_dataset_storage_structure.svg" width="820" alt="Dataset storage structure">
</p>

## Repository structure

```text
.
├── data/
│   ├── raw_measurements/          # 12 raw Cellular-Pro drive-test CSV files
│   ├── aligned_measurements/      # Measurements aligned with the local 3D scene
│   └── processed/                 # Analysis-ready PCI–RSRP tables
├── assets/                        # Terrain and building meshes
├── config/                        # Base-station, PCI and coordinate-alignment metadata
├── workflows/
│   ├── preprocessing/             # Coordinate alignment and measurement preprocessing
│   ├── parameter_calibration/     # Sionna RT parameter calibration
│   ├── radio_map/                 # Per-station and joint radio-map generation
│   ├── evaluation/                # Measurement–simulation comparison
│   ├── localization/              # Physical base-station localization
│   ├── reconstruction/            # Sparse radio-map reconstruction
│   └── visualization/             # Dataset and publication figures
├── tools/                         # Auxiliary repository utilities
├── docs/images/                   # Figures used in the paper and README
├── run_pipeline.py                # Unified workflow entry point
├── check_project_layout.py        # Repository/data-layout check
├── requirements.txt
└── environment.yml
```

Generated products are written to `outputs/` when the workflows are run.

## Dataset contents

| Component | Main contents | Role |
|---|---|---|
| Measurement data | 12 raw Cellular-Pro CSV files, aligned records, long-format PCI–RSRP observations, and 1 m processed tables | Field observations used for calibration, matching, localization, and reconstruction |
| 3D scene | `ground.ply` and `ynu_chenggong_campus-001.ply` | Terrain and major-building geometry used by Sionna RT |
| Physical base-station reference | 27 verified physical base stations and 79 associated PCIs | Links field measurements, physical sites, sectors, and simulated transmitters |
| Per-station radio maps | 1 m-grid radio maps for the 27 physical base stations | Sector/station-level radio propagation products |
| Joint best-server map | 4000 m × 3000 m, 1 m grid | Network-scale best-server RSRP, station, and PCI products |
| Matched measurement–simulation data | Co-located measured and simulated RSRP records | Direct evaluation of calibrated ray-tracing results |
| Application workflows | Two-branch localization and sparse reconstruction | Demonstrates paired use of measurement-only and measurement–simulation data |

## Environment

Python 3.10 is recommended. The ray-tracing workflows require Sionna RT and benefit substantially from a CUDA-capable NVIDIA GPU.

### Conda

```bash
conda env create -f environment.yml
conda activate sionna_env
```

### pip

```bash
python -m pip install -r requirements.txt
python -m pip install sionna==1.2.2
```

Run all commands from the repository root.

## Check the repository

```bash
python run_pipeline.py --help
python run_pipeline.py check
```

The layout check verifies the scene meshes, configuration files, PCI mapping, and measurement-data stage.

## Workflow

### 1. Prepare the drive-test measurements

To rebuild the processed measurement tables from the raw Cellular-Pro CSV files:

```bash
python run_pipeline.py prepare-data
```

To regenerate existing aligned files:

```bash
python run_pipeline.py prepare-data --force
```

The same workflow can also be executed step by step:

```bash
python run_pipeline.py align
python run_pipeline.py extract
python run_pipeline.py preprocess
```

Main processed products include:

```text
data/aligned_measurements/
data/processed/extracted/
data/processed/cell_pci_rsrp_long_27stations.csv
data/processed/cell_pci_rsrp_1m_calibration.csv
data/processed/cell_pci_rsrp_2p77m_localization.csv
```

The formal two-branch localization workflow reads:

```text
data/processed/cell_pci_rsrp_long_27stations.csv
```

The reconstruction workflow uses the 1 m processed measurement table.

### 2. Calibrate the physical base stations

A quick single-station run can be used to verify the Sionna RT environment:

```bash
python run_pipeline.py calibrate --stations 3 --quick
```

Run calibration for all 27 physical base stations:

```bash
python run_pipeline.py calibrate --stations all
```

The calibration workflow compares co-located measured and simulated RSRP for the same PCI and searches the configured base-station height, effective transmit power, azimuth, and downtilt parameters.

Main results:

```text
outputs/parameter_calibration/
├── all_27stations_summary.csv
├── estimated_initial_directions_27stations.csv
└── station_*/
```

### 3. Generate the per-station radio maps

```bash
python run_pipeline.py export-dem --stations all
```

The formal radio maps use a terrain-following receiver surface at **DEM + 1.5 m**, a 1 m horizontal grid, a maximum propagation depth of 5, and edge diffraction during radio-map generation.

Main output:

```text
outputs/bestparam_radio_maps_512m/
```

The dataset contains both original Sionna RT radio maps and measurement-filled radio maps. Measurement filling is applied only where the Sionna RT result is missing; existing valid simulation values are not overwritten.

### 4. Generate the joint best-server radio map

Optional checks before the full run:

```bash
python run_pipeline.py export-joint-map --dry-run
python run_pipeline.py export-joint-map --quick
```

Generate the full 27-station map:

```bash
python run_pipeline.py export-joint-map
```

The default network-scale map covers **4000 m × 3000 m** with a **1 m** grid and is processed in **48 tiles** of 500 m × 500 m.

Main results:

```text
outputs/joint_best_server_4000x3000/
├── joint_best_server_27stations_4000x3000.npz
├── joint_best_server_rsrp_4000x3000.png
├── joint_best_station_id_4000x3000.png
├── joint_best_pci_4000x3000.png
├── physical_station_best_server_area.csv
├── pci_best_server_area.csv
└── parameters_used_27stations_79sectors.csv
```

### 5. Compare the joint map with field measurements

```bash
python run_pipeline.py compare-joint-map
```

The workflow maps the road measurements to the common 1 m joint-radio-map grid and retains only grid cells with valid measured and simulated RSRP.

Main results:

```text
outputs/joint_map_measurement_comparison/
├── matched_measurement_vs_joint_map.csv
├── comparison_metrics.csv
├── comparison_metrics.json
├── comparison_diagnostics.json
├── comparison_summary.txt
└── comparison figures
```

The dataset article reports **10,383 valid matched grid samples** and a joint best-server RSRP RMSE of approximately **13.1 dB**.

### 6. Run the physical base-station localization example

Run the formal two-branch progressive localization experiment for all 27 physical base stations:

```bash
python run_pipeline.py localize-sweep \
  --point-counts 10,11,12,13,14,15 \
  --random-trials 10 \
  --random-seed 20260805 \
  --station-ids all
```

On Windows PowerShell, the same command can be entered on one line:

```powershell
python run_pipeline.py localize-sweep --point-counts 10,11,12,13,14,15 --random-trials 10 --random-seed 20260805 --station-ids all
```

The two branches use the **same receiver locations** within each physical station, receiver-count stage, and seeded trial:

- **Measurement-only:** estimates the base-station position from the selected road-measured PCI–RSRP observations using the robust profiled-RSS localization procedure.
- **Measurement–simulation:** uses the same measured observations and additionally incorporates co-located Sionna RT RSRP together with local spatial-gradient information sampled from the fixed PCI radio maps.

The receiver sets are strictly nested from 10 to 15 locations. Surveyed base-station coordinates are excluded from receiver selection, candidate construction, candidate scoring, progressive updating, and trial fusion; they are introduced only after the final station estimates are obtained to calculate localization error.

The unified entry point fixes the formal comparison to the two data-use branches and disables external direction priors.

Main output directory:

```text
outputs/localization_two_branch_rmse_only/
```

Files written by the current localization workflow:

```text
outputs/localization_two_branch_rmse_only/
├── localization_rmse_comparison.csv
├── localization_two_branch_trial_diagnostics.csv
├── measurement_only_station_results.csv
└── measurement_simulation_station_results.csv
```

The primary paper-level comparison is:

```text
localization_rmse_comparison.csv
```

with the columns:

```text
Receiver locations per station
Measurement-only RMSE (m)
Measurement–simulation RMSE (m)
```

The article reports:

| Receiver locations per station | Measurement-only RMSE (m) | Measurement–simulation RMSE (m) |
|---:|---:|---:|
| 10 | 198.1775 | 135.0829 |
| 11 | 189.0067 | 133.1364 |
| 12 | 180.5589 | 126.9115 |
| 13 | 174.3096 | 123.1994 |
| 14 | 165.5261 | 122.7997 |
| 15 | 160.4853 | 118.8830 |

### 7. Run the sparse radio-map reconstruction example

The paper example uses physical base station 3 and PCI 558 over a 512 m × 512 m, 1 m-grid region:

```bash
python run_pipeline.py reconstruct \
  --station-id 3 \
  --pci 558 \
  --percentages 1,2,3,4,5,6,7,8,9,10 \
  --selection-mode voronoi-safe-adaptive-nested \
  --simulation-mode compare \
  --random-seed 20260805
```

PowerShell one-line form:

```powershell
python run_pipeline.py reconstruct --station-id 3 --pci 558 --percentages 1,2,3,4,5,6,7,8,9,10 --selection-mode voronoi-safe-adaptive-nested --simulation-mode compare --random-seed 20260805
```

The two branches use the same strictly nested selected measurement sets:

- **Measurement-only:** 1-nearest-neighbor interpolation from selected measured RSRP.
- **Measurement–simulation:** robustly aligns the fixed Sionna RT map to the selected measurements and propagates the measured-minus-aligned-simulation residual using 1-nearest-neighbor assignment. Cells without valid Sionna RT values fall back to the measurement-only prediction.

The main evaluation is calculated over the complete valid outdoor grid of the Measurement-filled reference map.

Default output root:

```text
outputs/radio_map_reconstruction_nn_fullgrid_two_branch/
```

For the paper example:

```text
outputs/radio_map_reconstruction_nn_fullgrid_two_branch/
└── station_03_pci_558/
    ├── reference_filled_map/
    ├── percent_01/
    ├── ...
    ├── percent_10/
    ├── reconstruction_single_run_metrics.csv
    ├── reconstruction_simulation_ablation_comparison.csv
    ├── reconstruction_simulation_ablation_metrics.csv
    └── reconstruction_full_grid_evaluation_audit.csv
```

### 8. Generate dataset visualizations

```bash
python run_pipeline.py visualize-measurements
python run_pipeline.py plot-output-structure
```

## Example figures

### Campus 3D scene, drive-test measurements, and physical base stations

<p align="center">
  <img src="docs/images/figure3_drive_test_and_base_stations.png" width="900" alt="Drive-test measurements and physical base stations">
</p>

### Joint best-server RSRP and best-server PCI maps

<p align="center">
  <img src="docs/images/figure4_joint_best_server_radio_map.png" width="900" alt="Joint best-server radio map and best-server PCI map">
</p>

### Sparse radio-map reconstruction

<p align="center">
  <img src="docs/images/figure8_radio_map_reconstruction.png" width="900" alt="Radio-map reconstruction example">
</p>

## Data-generation summary

```text
Raw drive-test measurements
        ↓
WGS84 → EPSG:3857 → local Blender coordinate alignment
        ↓
DEM-based receiver elevation (local terrain + 1.5 m)
        ↓
Multi-PCI / RSRP expansion and spatial aggregation
        ↓
Field-verified 27-station / 79-PCI association
        ↓
Sionna RT base-station parameter calibration
        ↓
Per-station 1 m radio maps
        ↓
27-station joint best-server radio map
        ↓
Measurement–simulation spatial matching
        ↓
Localization and radio-map reconstruction examples
```

## Data availability
The dataset is available from Mendeley Data:
https://doi.org/10.17632/d5xrrgsj4f.1
