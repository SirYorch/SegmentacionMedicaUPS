import SimpleITK as sitk
import numpy as np
import cv2
import os
from typing import Dict, Any, Tuple

# Definición de rangos HU típicos para una CT de tórax (ajustables por el usuario final)
HU_RANGES = {
    "lung": (-1000, -300),    # Pulmón (Aire)
    "soft_tissue": (-100, 300), # Corazón/Tejido blando (Base)
    "bone": (300, 3000)       # Hueso
}

def load_and_extract_slice(file_path: str, slice_index: int = 0) -> Tuple[np.ndarray, sitk.Image]:
    """
    Carga un archivo de imagen médica (NIfTI, DICOM/IMA) usando SimpleITK.
    Si la imagen es 3D, extrae el slice, pero para archivos de un solo corte (como .ima),
    simplemente devuelve la imagen 2D cargada, ignorando el slice_index.
    """
    try:
        # Cargar la imagen usando SimpleITK (soporta NIfTI, DICOM, .ima, etc.)
        img_sitk = sitk.ReadImage(file_path)
        
        # =======================================================
        # LÓGICA DE SELECCIÓN DE SLICE 
        # =======================================================
        
        if img_sitk.GetDimension() == 3:
            total_slices = img_sitk.GetSize()[2] 
            
            if slice_index < 0 or slice_index >= total_slices:
                # Si el índice está fuera de rango, usamos el slice central
                center_slice = total_slices // 2
                print(f"Advertencia: Índice de slice ({slice_index}) fuera de rango (0 a {total_slices - 1}). Usando slice central: {center_slice}")
                slice_to_use = center_slice
            else:
                slice_to_use = slice_index
                
            # Extraer el slice 2D del volumen 3D
            img_sitk = img_sitk[:, :, slice_to_use]
        
        elif img_sitk.GetDimension() == 2:
            # Si es 2D (un solo corte, típico de .ima), aseguramos que el índice sea 0 para no confundir
            if slice_index != 0:
                print(f"Advertencia: La imagen es 2D (un solo corte). El índice {slice_index} se ignora.")
        
        # Si la dimensión es 2D, img_sitk ya es el slice.
        
        # Convertir la imagen ITK (HU) a un array NumPy (16 bits)
        slice_hu = sitk.GetArrayFromImage(img_sitk).astype(np.int16)
        
        return slice_hu, img_sitk
    
    except Exception as e:
        # Devolver None si la lectura o conversión falla
        print(f"Error CRÍTICO al cargar/extraer slice de la imagen: {e}")
        return None, None


def normalize_to_8bit(slice_hu: np.ndarray) -> np.ndarray:
    """
    Normaliza el array de HU (16 bits) a un rango de 8 bits (0-255) para visualización.
    """
    slice_8bit = cv2.normalize(slice_hu, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
    return slice_8bit


def apply_processing(slice_hu: np.ndarray, slice_8bit: np.ndarray, output_dir: str = "output_images") -> Dict[str, Any]:
    """
    Aplica la cadena de procesamiento: umbralización HU, refinamiento (Canny, Morfología) y resaltado.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    results = {
        "intermediate_images": {},
        "final_images": {},
        "extracted_masks": {},
        "metadata": {}
    }

    # 1. PASO INTERMEDIO: Canny para Detección de Bordes (en 8-bit)
    edges = cv2.Canny(slice_8bit, 50, 150)
    edges_dilated = cv2.dilate(edges, np.ones((5, 5), np.uint8), iterations=1)
    
    cv2.imwrite(os.path.join(output_dir, "01_edges_dilated.png"), edges_dilated)
    results["intermediate_images"]["edges_dilated"] = os.path.join(output_dir, "01_edges_dilated.png")

    # 2. Umbralización por Rangos HU (16-bit)
    lung_mask = cv2.inRange(slice_hu, HU_RANGES["lung"][0], HU_RANGES["lung"][1])
    heart_mask = cv2.inRange(slice_hu, HU_RANGES["soft_tissue"][0], HU_RANGES["soft_tissue"][1])
    bone_mask = cv2.inRange(slice_hu, HU_RANGES["bone"][0], HU_RANGES["bone"][1])

    cv2.imwrite(os.path.join(output_dir, "02_lung_raw_mask.png"), lung_mask)
    cv2.imwrite(os.path.join(output_dir, "03_heart_raw_mask.png"), heart_mask)
    cv2.imwrite(os.path.join(output_dir, "04_bone_raw_mask.png"), bone_mask)
    
    results["extracted_masks"]["lung_raw"] = os.path.join(output_dir, "02_lung_raw_mask.png")
    results["extracted_masks"]["heart_raw"] = os.path.join(output_dir, "03_heart_raw_mask.png")
    results["extracted_masks"]["bone_raw"] = os.path.join(output_dir, "04_bone_raw_mask.png")


    # 3. Refinamiento Morfológico y Eliminación de Bordes
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    
    lung_mask_refined = cv2.morphologyEx(lung_mask, cv2.MORPH_OPEN, kernel)
    heart_mask_refined = cv2.morphologyEx(heart_mask, cv2.MORPH_OPEN, kernel)
    bone_mask_refined = cv2.morphologyEx(bone_mask, cv2.MORPH_CLOSE, kernel)

    inv_edges = cv2.bitwise_not(edges_dilated)

    lung_mask_refined   = cv2.bitwise_and(lung_mask_refined, inv_edges)
    heart_mask_refined  = cv2.bitwise_and(heart_mask_refined, inv_edges)
    bone_mask_refined   = cv2.bitwise_and(bone_mask_refined, inv_edges)

    cv2.imwrite(os.path.join(output_dir, "05_lung_refined_mask.png"), lung_mask_refined)
    cv2.imwrite(os.path.join(output_dir, "06_heart_refined_mask.png"), heart_mask_refined)
    cv2.imwrite(os.path.join(output_dir, "07_bone_refined_mask.png"), bone_mask_refined)

    results["extracted_masks"]["lung_refined"] = os.path.join(output_dir, "05_lung_refined_mask")
    results["extracted_masks"]["heart_refined"] = os.path.join(output_dir, "06_heart_refined_mask")
    results["extracted_masks"]["bone_refined"] = os.path.join(output_dir, "07_bone_refined_mask")
    
    # 4. Resaltado de Color sobre la Imagen Base
    
    base_color = cv2.cvtColor(slice_8bit, cv2.COLOR_GRAY2BGR)
    highlighted_image = base_color.copy()

    highlighted_image[lung_mask_refined > 0] = (255, 255, 0) # Pulmón (Cian)
    highlighted_image[heart_mask_refined > 0] = (0, 0, 255)  # Corazón/Tejido blando (Rojo)
    highlighted_image[bone_mask_refined > 0] = (0, 255, 0)   # Hueso (Verde)
    
    final_path = os.path.join(output_dir, "12_final_highlighted.png")
    cv2.imwrite(final_path, highlighted_image)
    results["final_images"]["highlighted_roi"] = final_path

    cv2.imwrite(os.path.join(output_dir, "00_slice_gray.png"), slice_8bit)
    results["intermediate_images"]["slice_8bit"] = os.path.join(output_dir, "00_slice_gray.png")


    _, buffer = cv2.imencode('.png', highlighted_image)
    results["metadata"]["highlighted_base64"] = buffer.tobytes().decode('latin1')
    
    return results

if __name__ == '__main__':
    print("Módulo de procesamiento de imágenes médicas cargado.")