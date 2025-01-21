#create the folder if it does not exist
mkdir -p ~/Documents/Geom_3D_CV_proj/results/RBF
mkdir -p ~/Documents/Geom_3D_CV_proj/results/RBF_cuda

#download the files
scp -r michele@lottohouse.duckdns.org:/home/michele/Documents/Geom_3D_CV_proj/results/RBF ~/Documents/Geom_3D_CV_proj/results/RBF
scp -r michele@lottohouse.duckdns.org:/home/michele/Documents/Geom_3D_CV_proj/results/RBF_cuda ~/Documents/Geom_3D_CV_proj/results/RBF_cuda