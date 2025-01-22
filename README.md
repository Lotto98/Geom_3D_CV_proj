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

### Usage Instructions for `calibrator.py`

This script is used to calibrate camera using chessboard images.

```bash
usage: calibration.py [-h] [--name NAME] [--chessboard_dim CHESSBOARD_DIM] [--num_frames NUM_FRAMES] 
                        [--debug | --no-debug] [--error | --no-error]
```

#### Options

| Option                            | Description                                                 |
|-----------------------------------|-------------------------------------------------------------|
| `-h, --help`                      | Show this help message and exit.                            |
| `--name NAME`                     | Name of the camera setup (moving_light/static).             |
| `--chessboard_dim CHESSBOARD_DIM` | Chessboard dimensions.                                      |
| `--num_frames NUM_FRAMES`         | Number of frames to select from the video.                  |
| `--debug, --no-debug`             | Show the chessboard corners and projected corners.          |
| `--error, --no-error`             | Calculate the error.                                        |


### Usage Instructions for `analysis.py`

This script is used to compute the light for a given coin and interpolate it using different methods.

```bash
usage: analysis.py [-h] --coin {1,2,3,4} [--compute-light] [--interpolate] 
                    [--debug] [--debug-moving] [--debug-static] 
                    [--method {RBF,PTM,RBF_cuda}] [--nprocesses NPROCESSES]
                    [--coin-dim WIDTH HEIGHT] [--regular-grid-dim ROWS COLS]
```

#### Options

| Option                        | Description                                                                                        |
|-------------------------------|----------------------------------------------------------------------------------------------------|
| `-h, --help`                  | Show this help message and exit.                                                                   |
| `--coin {1,2,3,4}`            | Specify the coin type to process (1, 2, 3, or 4).                                                  |
| `--compute-light`             | Enable light computation.                                                                          |
| `--interpolate`               | Enable interpolation for computed light.                                                           |
| `--debug`                     | Enable debug mode for general processing. For `--compute-light` only.                              |
| `--debug-moving`              | Enable debug mode for moving light. For `--compute-light` only.                                    |
| `--debug-static`              | Enable debug mode for static light. For `--compute-light` only.                                    |
| `--method {RBF,PTM,RBF_cuda}` | Specify the interpolation method. For `--interpolate` only.                                        |
| `--nprocesses NPROCESSES`     | Number of processes for interpolation (Default: -1 to use all available). For `--method RBF` only. |
| `--coin-dim WIDTH HEIGHT`     | Dimensions of the coin (default: `[512, 512]`). For `--interpolate` only.                          |
| `--regular-grid-dim ROWS COLS`| Dimensions of the regular grid (default: `[100, 100]`). For `--interpolate` only.                  |

#### Example Usage

##### Basic Example
```bash
python analysis.py --coin 1 --compute-light --debug
```

##### Interpolation with Custom Coin and Grid Dimensions
```bash
python analysis.py --coin 3 --interpolate --coin-dim 256 256 --regular-grid-dim 50 50 --method RBF_cuda
```

### Usage `relighting.py`

This script relight a coin using the data from a specified interpolation method.

```bash
usage: relighting.py [-h] --coin COIN --method {RBF,PTM,RBF_cuda} 
                            --coin-dim COIN_DIM COIN_DIM 
                            --regular-grid-dim REGULAR_GRID_DIM REGULAR_GRID_DIM
```

#### Options

| Option                                                 | Description                                                            |
|--------------------------------------------------------|------------------------------------------------------------------------|
| `-h, --help`                                           | Show this help message and exit.                                       |
| `--coin COIN`                                          | Coin number: 1, 2, 3, 4.                                               |
| `--method {RBF,PTM,RBF_cuda}`                          | Method of interpolation to visualize.                                  |
| `--coin-dim COIN_DIM COIN_DIM`                         | Coin dimensions.                                                       |
| `--regular-grid-dim REGULAR_GRID_DIM REGULAR_GRID_DIM` | Regular grid dimensions.                                               |

#### Example Usage

Relight coin 1 with method `RBF_cuda`:

```bash
python3 relighting.py --coin 1 --method RBF_cuda --coin-dim 512 512 --regular-grid-dim 100 100 
```

### Usage `interpolation_visualizer.py`
