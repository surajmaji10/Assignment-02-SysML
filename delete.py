import os
import glob

def delete_files(extensions):
    for ext in extensions:
        files = glob.glob(f'**/*{ext}', recursive=True)
        for file in files:
            try:
                os.remove(file)
                print(f"Deleted: {file}")
            except Exception as e:
                print(f"Error deleting {file}: {e}")

if __name__ == "__main__":
    extensions_to_delete = [".txt", ".o", ".out", ".exe"]
    delete_files(extensions_to_delete)
