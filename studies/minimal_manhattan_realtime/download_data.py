import os
import sys
import shutil
import zipfile
import urllib.request

ZENODO_URL = "https://zenodo.org/api/records/15187906/files/FleetPy_Manhattan.zip/content"

def download_and_extract_manhattan_data():
    fleetpy_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    data_dir = os.path.join(fleetpy_root, "data")
    zip_path = os.path.join(data_dir, "FleetPy_Manhattan.zip")

    os.makedirs(data_dir, exist_ok=True)

    print(f"Downloading Manhattan dataset (~408 MB) from Zenodo...")
    print(f"URL: {ZENODO_URL}")
    print(f"Target zip: {zip_path}\n")

    def progress_hook(count, block_size, total_size):
        downloaded = count * block_size
        if total_size > 0:
            percent = downloaded / total_size * 100
            mb_downloaded = downloaded / (1024 * 1024)
            mb_total = total_size / (1024 * 1024)
            sys.stdout.write(f"\rProgress: {percent:.1f}% ({mb_downloaded:.1f} / {mb_total:.1f} MB)")
            sys.stdout.flush()

    try:
        # Step 1: Download zip archive
        urllib.request.urlretrieve(ZENODO_URL, zip_path, reporthook=progress_hook)
        print("\n\nDownload complete! Extracting files...")
        
        # Step 2: Unzip into data directory
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(data_dir)
            
        # Step 3: Auto-organize nested subdirectories (e.g. data/FleetPy_Manhattan/ -> data/)
        wrapper_dir = os.path.join(data_dir, "FleetPy_Manhattan")
        if os.path.exists(wrapper_dir):
            print("Organizing extracted subdirectories into FleetPy/data/...")
            for category in ["networks", "demand", "zones", "infra", "pubtrans", "vehicles"]:
                category_src = os.path.join(wrapper_dir, category)
                category_dst = os.path.join(data_dir, category)
                if os.path.exists(category_src):
                    os.makedirs(category_dst, exist_ok=True)
                    for item in os.listdir(category_src):
                        item_src = os.path.join(category_src, item)
                        item_dst = os.path.join(category_dst, item)
                        if not os.path.exists(item_dst):
                            shutil.move(item_src, item_dst)
                            print(f"  - Moved data/{category}/{item}")
                        else:
                            print(f"  - Already exists: data/{category}/{item}")
            shutil.rmtree(wrapper_dir)

        # Step 4: Clean up downloaded zip file
        if os.path.exists(zip_path):
            os.remove(zip_path)
            print("Cleaned up temporary zip archive.")

        print("\n✅ Manhattan dataset successfully downloaded and organized under FleetPy/data/")
    except Exception as e:
        print(f"\n❌ Error downloading/organizing data: {e}")

if __name__ == "__main__":
    download_and_extract_manhattan_data()
