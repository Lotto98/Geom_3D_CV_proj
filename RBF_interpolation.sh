eval "$(conda shell.bash hook)"
conda activate Geom_3D_CV

python3 src/2_analysis.py --coin 1 --interpolate --method RBF --nprocesses -1
python3 src/2_analysis.py --coin 2 --interpolate --method RBF --nprocesses -1
python3 src/2_analysis.py --coin 3 --interpolate --method RBF --nprocesses -1
python3 src/2_analysis.py --coin 4 --interpolate --method RBF --nprocesses -1