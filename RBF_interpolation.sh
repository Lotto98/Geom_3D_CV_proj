eval "$(conda shell.bash hook)"
conda activate Geom_3D_CV

python3 src/analysis.py --coin 1 --interpolate --method RBF_cuda
python3 src/analysis.py --coin 2 --interpolate --method RBF_cuda 
python3 src/analysis.py --coin 3 --interpolate --method RBF_cuda
python3 src/analysis.py --coin 4 --interpolate --method RBF_cuda