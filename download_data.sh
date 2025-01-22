
# Download the data
wget https://www.dsi.unive.it/~bergamasco/teachingfiles/G3DCV2024_data.7z

archive_path="G3DCV2024_data.7z"
output_folder="data"

# Check if the 7z archive exists
if [[ ! -f "$archive_path" ]]; then
    echo "Error: File '$archive_path' does not exist."
    exit 1
fi

# Extract the archive
echo "Extracting $archive_path to $output_folder..."
7z x "$archive_path"

#Rename the folder
mv G3DCV2024_data "$output_folder"

#Delete the archive
rm G3DCV2024_data.7z