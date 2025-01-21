# Geom_3D_CV_proj

## Installation

To set up the conda environment for this project, follow these steps:

1. **Clone the repository:**
    ```bash
    git clone https://github.com/yourusername/Geom_3D_CV_proj.git
    cd Geom_3D_CV_proj
    ```

2. **Create the conda environment:**
    ```bash
    conda env create -f environment.yml
    ```

3. **Activate the conda environment:**
    ```bash
    conda activate geom_3D_CV
    ```

## Usage

### Calibrator

To run the `calibrator.py` script, use the following command:
```bash
usage: calibration.py [-h] [--name NAME] [--chessboard_dim CHESSBOARD_DIM] [--num_frames NUM_FRAMES] [--debug | --no-debug] [--error | --no-error]

Calibrate camera using chessboard images.

options:
  -h, --help            show this help message and exit
  --name NAME           Name of the camera setup (moving_light/static)
  --chessboard_dim CHESSBOARD_DIM
                        Chessboard dimensions
  --num_frames NUM_FRAMES
                        Number of frames to select from the video
  --debug, --no-debug   Show the chessboard corners and projected corners
  --error, --no-error   Calculate the error
```

### Analysis

To run the `analysis.py` script, use the following command:
```bash
python analysis.py --data <data_file> --results <results_file>
```
Replace `<data_file>` with the path to your data file and `<results_file>` with the desired path for the results file.

### Relighting

To run the `relighting.py` script, use the following command:
```bash
python relighting.py --scene <scene_file> --output <output_file>
```
Replace `<scene_file>` with the path to your scene file and `<output_file>` with the desired path for the output file.