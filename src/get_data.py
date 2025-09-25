import os

def download_data():
    """
    Download dataset from Kaggle using Kaggle API.
    Requires kaggle.json in ~/.kaggle/
    """
    os.makedirs("data", exist_ok=True)

    # Dataset de ejemplo (puedes cambiarlo por otro de Kaggle)
    dataset = "colearninglounge/hr-analytics-job-change-of-data-scientists"

    print(f"Descargando dataset {dataset}...")
    os.system(f"kaggle datasets download -d {dataset} -p data/ --unzip")
    print("✅ Descarga completada. Archivos en carpeta 'data/'.")

if __name__ == "__main__":
    download_data()
