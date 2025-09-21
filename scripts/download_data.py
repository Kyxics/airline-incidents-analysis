import kagglehub
import shutil
import os
from pathlib import Path

def download_airline_data():
	# Download latest version
	download_path = kagglehub.dataset_download("tarique7/airline-incidents-safety-data")
	print("Downloaded to: ", download_path)

	# Copy to data folder
	download_dir = Path(download_path)
	data_dir = Path("../data")

	# Create dir
	try:
		data_dir.mkdir(exist_ok=True)
	except Exception as e:
		print(f"Error cereating directory: {e}")
		return None

	# Copy to project data folder
	for file_path in download_dir.iterdir():
		if file_path.is_file():
			destination = data_dir / file_path.name
			shutil.copy(file_path, destination)
			print(f"Copied {file_path.name} to {data_dir} folder.")

	return data_dir.resolve()

if __name__ == "__main__":
	data_path = download_airline_data()
	print("Data ready in: ", data_path)
