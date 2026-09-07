# Campus-Scale 5G NR Radio Propagation Dataset

This repository contains the data-processing and application code associated with a campus-scale 5G NR radio-propagation dataset collected at the Chenggong Campus of Yunnan University, Kunming, China.

The dataset combines **vehicle-based 5G NR measurements**, **3D terrain and building geometry**, **27 physical base stations with 79 PCIs**, and **Sionna RT radio maps** in a common spatial coordinate system. It also includes application workflows for measurement–simulation comparison, base-station localization, and sparse radio-map reconstruction.

<p align="center">
  <img src="docs/images/figure1_dataset_storage_structure.png" width="820" alt="Dataset structure">
</p>

## Repository structure

```text
.
├── data/                       # Measurement data and processed measurement tables
├── assets/                     # 3D terrain and building meshes
├── config/                     # Base-station, PCI and coordinate-alignment information
├── workflows/
│   ├── preprocessing/          # Coordinate alignment and measurement preprocessing
│   ├── parameter_calibration/  # Sionna RT parameter calibration
│   ├── radio_map/              # Per-station and joint radio-map generation
│   ├── evaluation/             # Measurement–simulation comparison
│   ├── localization/           # Physical base-station localization
│   ├── reconstruction/         # Sparse radio-map reconstruction
│   └── visualization/          # Dataset and publication figures
├── scripts/windows/            # Optional Windows batch wrappers
├── docs/images/                # Figures used in the paper and README
├── tools/                      # Auxiliary repository tools
└── outputs/                    # Generated results after running the workflows
```

## Main components

| Part | Main contents | Purpose |
|---|---|---|
| `data/` | 12 raw Cellular-Pro CSV files, aligned measurements and processed PCI–RSRP tables | Provides the field-measurement observations used by all downstream workflows |
| `assets/` | `ground.ply` and campus building mesh | Provides the terrain and building geometry used by Sionna RT |
| `config/` | Physical base-station catalogue, PCI/site mapping and coordinate-alignment metadata | Connects measurements, physical sites and the 3D scene |
| `workflows/preprocessing/` | Coordinate conversion, DEM height extraction, multi-PCI expansion and spatial aggregation | Converts raw drive-test files into analysis-ready measurement tables |
| `workflows/parameter_calibration/` | Sionna RT calibration code | Estimates station/sector parameters by comparing simulation with field measurements |
| `workflows/radio_map/` | Single-station and network-scale radio-map generation | Produces per-station radio maps and the joint best-server map |
| `workflows/evaluation/` | Measurement–simulation matching and error calculation | Evaluates the joint radio map on co-located measurement samples |
| `workflows/localization/` | Measurement-only and measurement–simulation localization | Provides the physical base-station localization application example |
| `workflows/reconstruction/` | Measurement-only and measurement–simulation reconstruction | Provides the sparse radio-map reconstruction application example |
| `workflows/visualization/` | Dataset overview and publication plotting scripts | Generates figures used to inspect and present the dataset |

## Installation

Python 3.10 is recommended. Install the main dependencies with:

```bash
pip install numpy pandas scipy matplotlib trimesh pyproj shapely scikit-learn pyyaml torch sionna
```

A CUDA-capable NVIDIA GPU is recommended for the full Sionna RT calibration and radio-map generation workflows.

## Workflow

Run all Python commands from the repository root.

### 1. Prepare the measurements

```bash
python workflows/preprocessing/01_align_measurements_to_blender.py
python workflows/preprocessing/02_extract_multi_pci_rsrp.py
python workflows/preprocessing/03_build_analysis_tables.py
```

`01` aligns raw drive-test coordinates with the Blender scene and adds terrain/receiver height. `02` expands all PCI–RSRP pairs. `03` aggregates the aligned records into analysis-ready tables.

Main results:

```text
data/aligned_measurements/
data/processed/extracted/
data/processed/cell_pci_rsrp_long_27stations.csv
data/processed/cell_pci_rsrp_1m_calibration.csv
data/processed/cell_pci_rsrp_2p77m_localization.csv
```

### 2. Calibrate the physical base stations

```bash
python workflows/parameter_calibration/run_27station_parameter_calibration.py
```

Runs calibration for all physical stations, compares Sionna RT predictions with field measurements, and saves the selected station/sector calibration results and summaries.

Main results:

```text
outputs/parameter_calibration/
├── all_27stations_summary.csv
├── estimated_initial_directions_27stations.csv
└── station_*/
```

### 3. Generate the radio maps

Per-station radio maps:

```bash
python workflows/radio_map/export_bestparam_radio_maps.py
```

Uses the calibrated station parameters to export the per-station radio-map products used by the dataset.

Network-scale joint best-server map:

```bash
python workflows/radio_map/generate_joint_best_server_4000x3000.py
```

Combines all station products on one grid and generates the network-scale best-server RSRP, serving-station, and best-server PCI maps.

Main results:

```text
outputs/bestparam_radio_maps_512m/
outputs/joint_best_server_4000x3000/
├── joint_best_server_27stations_4000x3000.npz
├── joint_best_server_rsrp_4000x3000.png
├── joint_best_station_id_4000x3000.png
└── joint_best_pci_4000x3000.png
```

### 4. Compare the joint map with field measurements

```bash
python workflows/evaluation/compare_joint_map_with_measurements.py
```

Matches field samples to the joint radio-map grid, extracts simulated values at the same locations, and reports measurement–simulation comparison metrics and figures.

Main results:

```text
outputs/joint_map_measurement_comparison/
├── matched_measurement_vs_joint_map.csv
├── comparison_metrics.csv
└── comparison figures
```

### 5. Run the localization example

```bash
python workflows/localization/run_27stations_two_branch_localization.py
```

Runs the localization application for the 27 physical stations using the measurement-only and measurement–simulation branches and exports their RMSE comparison.

Main result:

```text
outputs/localization_two_branch_rmse_only/localization_rmse_comparison.csv
```

### 6. Run the radio-map reconstruction example

```bash
python workflows/reconstruction/run_reconstruction.py
```

Reconstructs radio maps from progressively sampled measurements, evaluates measurement-only and measurement–simulation branches, and saves reconstruction metrics and result figures.

Main results are written to:

```text
outputs/radio_map_reconstruction_nn_single_progressive/
```

The reconstruction workflow compares the **measurement-only** and **measurement–simulation** branches over progressively increasing measurement sampling ratios.

## Example results from the dataset

### Drive-test measurements and physical base stations

<p align="center">
  <img src="docs/images/figure3_drive_test_and_base_stations.png" width="900" alt="Drive-test measurements and physical base stations">
</p>

### Joint best-server radio map for the 27 physical base stations

<p align="center">
  <img src="docs/images/figure4_joint_best_server_radio_map.png" width="900" alt="Joint best-server radio map">
</p>

### Best-server PCI in the joint radio map

<p align="center">
  <img src="docs/images/figure5_joint_best_server_pci.png" width="900" alt="Joint best-server PCI map">
</p>

### Sparse radio-map reconstruction

<p align="center">
  <img src="docs/images/figure8_radio_map_reconstruction.png" width="900" alt="Radio-map reconstruction example">
</p>

## Dataset workflow summary

```text
Raw drive-test measurements
        ↓
Coordinate alignment + DEM height extraction
        ↓
PCI–RSRP expansion and measurement aggregation
        ↓
Sionna RT parameter calibration
        ↓
Per-station radio maps
        ↓
27-station joint best-server radio map
        ↓
Measurement–simulation evaluation
        ↓
Localization and radio-map reconstruction examples
```
