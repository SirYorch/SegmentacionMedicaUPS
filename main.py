import uvicorn
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
# Importar el módulo para servir archivos estáticos
from fastapi.staticfiles import StaticFiles 
# Importar el middleware CORS
from fastapi.middleware.cors import CORSMiddleware 
from med_image_processor import load_and_extract_slice, normalize_to_8bit, apply_processing, HU_RANGES
import os
import shutil
from typing import Dict, Any

# =======================================================
# CONFIGURACIÓN DE RUTAS Y DIRECTORIOS
# =======================================================
app = FastAPI(
    title="API de Procesamiento de Imágenes Médicas (FastAPI)",
    description="Backend para el Proyecto de Visión Artificial - Extracción de ROI de CT Scans.",
    version="1.0.0"
)

# =======================================================
# CONFIGURACIÓN CORS (CRUCIAL PARA QUE EL FRONTEND FUNCIONE)
# =======================================================
# Definir los orígenes que pueden acceder a tu API.
# "*" permite CUALQUIER origen, lo cual es útil para pruebas locales.
origins = [
    "http://localhost",
    "http://localhost:8000",
    "http://127.0.0.1:8000",
    "file://*", # NECESARIO si abres index.html directamente
    "*"          # Permite cualquier origen (usar con cuidado en producción)
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"], # Permitir todos los métodos (GET, POST, etc.)
    allow_headers=["*"], # Permitir todos los encabezados
)
# =======================================================


# Directorios temporales y de salida
UPLOAD_DIR = "temp_uploads"
OUTPUT_DIR = "processed_data"

if not os.path.exists(UPLOAD_DIR):
    os.makedirs(UPLOAD_DIR)
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

# MONTAR DIRECTORIO ESTÁTICO
app.mount("/static", StaticFiles(directory=OUTPUT_DIR), name="static")

# =======================================================
# ENDPOINTS (Sin cambios en la lógica)
# =======================================================

@app.get("/", response_class=HTMLResponse, tags=["Información"])
async def root():
    """
    Endpoint de bienvenida que sirve el frontend HTML.
    """
    try:
        return HTMLResponse("""
        <!DOCTYPE html>
        <html>
        <head><meta charset="UTF-8"><meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>Visor Médico Inteligente - Frontend</title>
        </head>
        <body>
            <p>Tu frontend HTML debe ser servido aquí, o accede a la documentación en <a href="/docs">/docs</a>.</p>
            <p><strong>Recuerda:</strong> Asegúrate de ejecutar el servidor con `uvicorn main:app --reload` y cargar tu `index.html` en el navegador.</p>
        </body>
        </html>
        """)
        
    except Exception as e:
        return HTMLResponse(f"Error al cargar el HTML de inicio: {e}", status_code=500)


@app.post("/uploadfile/", tags=["Archivos"])
async def upload_file(file: UploadFile = File(...)) -> Dict[str, str]:
    """
    Sube un archivo de imagen médica (NIfTI/DICOM) al servidor temporal.
    """
    file_path = os.path.join(UPLOAD_DIR, file.filename)
    try:
        # Asegurarse de que el directorio de salida del procesamiento esté limpio para este archivo
        process_output_path = os.path.join(OUTPUT_DIR, os.path.splitext(file.filename)[0])
        if os.path.exists(process_output_path):
             # Eliminar el directorio si existe para limpiar resultados anteriores
             shutil.rmtree(process_output_path)
             os.makedirs(process_output_path)
        
        # Guardar el archivo subido
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error al guardar el archivo: {e}")

    return {"message": f"Archivo '{file.filename}' subido con éxito.", "filename": file.filename, "path": file_path}

@app.get("/process/{filename}", tags=["Procesamiento"])
async def process_image(filename: str, slice_index: int = 0) -> JSONResponse:
    """
    Procesa un slice de la imagen subida, aplica segmentación y devuelve los resultados.
    Genera y guarda imágenes intermedias y finales en el directorio 'processed_data/{filename}/'.
    """
    file_path = os.path.join(UPLOAD_DIR, filename)
    base_name = os.path.splitext(filename)[0]
    
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="Archivo no encontrado. Asegúrate de haberlo subido primero.")

    # 1. Cargar y extraer el slice
    slice_hu, img_sitk = load_and_extract_slice(file_path, slice_index)

    if slice_hu is None:
         raise HTTPException(status_code=500, detail="Error en la lectura o extracción del slice de la imagen.")
    
    # 2. Normalizar a 8 bits
    slice_8bit = normalize_to_8bit(slice_hu)
    
    # 3. Aplicar la cadena de procesamiento
    process_output_dir = os.path.join(OUTPUT_DIR, base_name)
    results = apply_processing(slice_hu, slice_8bit, output_dir=process_output_dir)

    # 4. Preparar la respuesta JSON
    
    # Modificamos las rutas a URLs para que el frontend pueda acceder a ellas
    def map_to_url(path):
        # Reemplaza la ruta local (processed_data/nombre_archivo/...) con la URL estática (/static/nombre_archivo/...)
        return path.replace(OUTPUT_DIR, "/static").replace("\\", "/") # Para compatibilidad con Windows/Linux
        
    url_results = {
        "intermediate_images": {k: map_to_url(v) for k, v in results["intermediate_images"].items()},
        "extracted_masks": {k: map_to_url(v) for k, v in results["extracted_masks"].items()},
        "final_image": map_to_url(results["final_images"]["highlighted_roi"])
    }
    
    response_data = {
        "filename": filename,
        "slice_index_processed": slice_index,
        "hu_ranges_used": HU_RANGES,
        "image_highlighted_base64": results["metadata"].get("highlighted_base64", ""),
        "saved_files_urls": url_results,
        "message": "Procesamiento completado. Máscaras y resaltado de áreas de interés generados."
    }
    
    return JSONResponse(content=response_data)