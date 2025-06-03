import os
import shutil
import xml.etree.ElementTree as ET
from kaggle.api.kaggle_api_extended import KaggleApi

def download_dataset(kaggle_json_path, download_path="data"):
    os.environ['KAGGLE_CONFIG_DIR'] = os.path.dirname(kaggle_json_path)
    api = KaggleApi()
    api.authenticate()
    api.dataset_download_files("andrewmvd/face-mask-detection", path=download_path, unzip=True)
    print("Dataset downloaded and extracted to", download_path)
    
def organize_dataset(
    annotations_dir='data/annotations',
    images_dir='data/images',
    output_dir='dataset'
):
    labels_map = {
        'with_mask': 'with_mask',
        'without_mask': 'without_mask',
        'mask_weared_incorrect': 'mask_weared_incorrect'
    }

    # Buat folder kelas jika belum ada
    for label in labels_map.values():
        os.makedirs(os.path.join(output_dir, label), exist_ok=True)

    # Baca file XML dan pindahkan gambar ke folder kelas sesuai label
    for xml_file in os.listdir(annotations_dir):
        if not xml_file.endswith('.xml'):
            continue

        xml_path = os.path.join(annotations_dir, xml_file)
        tree = ET.parse(xml_path)
        root = tree.getroot()

        filename = root.find('filename').text
        label = root.find('object').find('name').text

        if label in labels_map:
            src_image_path = os.path.join(images_dir, filename)
            dst_image_path = os.path.join(output_dir, labels_map[label], filename)

            if os.path.exists(src_image_path):
                shutil.copy(src_image_path, dst_image_path)

    print("✅ Dataset telah dipindahkan ke folder per kelas di:", output_dir)