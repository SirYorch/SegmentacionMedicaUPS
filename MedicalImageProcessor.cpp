#include "MedicalImageProcessor.h"
#include <iomanip> // For std::setprecision

// Definición de tipos ITK para un solo slice 2D
using PixelType  = short;                 
constexpr unsigned int Dimension = 2;     
using ImageType  = itk::Image<PixelType, Dimension>;
using ReaderType = itk::ImageFileReader<ImageType>;

// ====================================================================
// CONSTRUCTOR y Métodos de Carga (loadAndConvertSlice)
// ====================================================================

// Constructor
MedicalImageProcessor::MedicalImageProcessor(const std::string& filePath, const std::string& outputDirectory)
    : inputFilePath(filePath), outputDir(outputDirectory) {
    // Asegurar que el directorio de salida exista
    fs::create_directories(outputDir);
    std::cout << "Directorio de salida creado/verificado: " << outputDir << std::endl;
}

/**
 * Implementación de la carga y conversión ITK -> OpenCV
 */
bool MedicalImageProcessor::loadAndConvertSlice(const std::string& filePath) {
    auto reader = ReaderType::New();
    reader->SetFileName(filePath);

    try {
        // La actualización realiza la lectura y la convierte a la dimensión 2D forzada
        reader->Update();
    } catch (itk::ExceptionObject &ex) {
        std::cerr << "Error CRÍTICO leyendo el archivo ITK: " << ex << std::endl;
        return false;
    }

    ImageType::Pointer image = reader->GetOutput();
    ImageType::RegionType region = image->GetLargestPossibleRegion();
    ImageType::SizeType size   = region.GetSize();

    std::cout << "Dimensiones imagen: " << size[0] << " x " << size[1] << std::endl;

    // ITK -> OpenCV (Mat de 16 bits, HU original)
    cvSliceHU.create(static_cast<int>(size[1]), 
                     static_cast<int>(size[0]), 
                     CV_16SC1); // CV_16SC1 para short (HU)

    itk::ImageRegionConstIterator<ImageType> it(image, region);

    for (unsigned int y = 0; y < size[1]; ++y) {
        for (unsigned int x = 0; x < size[0]; ++x) {
            ImageType::IndexType idx;
            idx[0] = x;
            idx[1] = y;
            it.SetIndex(idx);

            cvSliceHU.at<short>(y, x) = it.Get();  // HU original
        }
    }
    
    // Preparar el slice 8bit de base (para visualización en la GUI)
    cvSlice8bit.create(cvSliceHU.size(), CV_8UC1);
    cv::normalize(cvSliceHU, cvSlice8bit, 0, 255, cv::NORM_MINMAX);
    cvSlice8bit.convertTo(cvSlice8bit, CV_8UC1);
    
    // Guardar imagen base 8-bit (Paso intermedio 00 - Evidencia)
    fs::path savePath = fs::path(outputDir) / "00_slice_gray_base.png";
    cv::imwrite(savePath.string(), cvSlice8bit);
    std::cout << "Guardado: " << savePath.string() << std::endl;
    
    return true;
}

// ====================================================================
// Implementación de la Cadena de Procesamiento (Llamada por executeFinalProcessing)
// ====================================================================

/**
 * Implementación de la cadena de procesamiento y guardado de máscaras
 */
void MedicalImageProcessor::applyProcessingChain(cv::Mat& lungMask, cv::Mat& heartMask, cv::Mat& boneMask) {
    // NOTA: Aquí se usan los rangos HU del mapa HU_RANGES, que la GUI actualizó antes de llamar a esta función.

    // =========================================================
    // PASO 1: DETECCIÓN DE BORDES (CANNY) y DILATACIÓN
    // =========================================================
    cv::Mat edges;
    cv::Canny(cvSlice8bit, edges, 50, 150); // Usamos la imagen 8-bit normalizada para Canny

    cv::Mat edgesDilated;
    cv::dilate(edges, edgesDilated, cv::Mat(), cv::Point(-1,-1), 1);
    
    fs::path savePath_edges = fs::path(outputDir) / "01_edges_dilated.png";
    cv::imwrite(savePath_edges.string(), edgesDilated);
    std::cout << "Guardado: " << savePath_edges.string() << std::endl;


    // =========================================================
    // PASO 2: UMBRALIZACIÓN POR RANGOS (HU) con valores AJUSTADOS
    // =========================================================
    // Umbralización para Pulmón
    cv::inRange(cvSliceHU,
                cv::Scalar(HU_RANGES.at("lung").min),
                cv::Scalar(HU_RANGES.at("lung").max),
                lungMask);

    // Umbralización para Tejido Blando/Corazón
    cv::inRange(cvSliceHU,
                cv::Scalar(HU_RANGES.at("soft_tissue").min),
                cv::Scalar(HU_RANGES.at("soft_tissue").max),
                heartMask);

    // Umbralización para Hueso
    cv::inRange(cvSliceHU,
                cv::Scalar(HU_RANGES.at("bone").min),
                cv::Scalar(HU_RANGES.at("bone").max),
                boneMask);

    fs::path savePath_lung_raw = fs::path(outputDir) / "02_lung_raw_mask.png";
    cv::imwrite(savePath_lung_raw.string(), lungMask);
    fs::path savePath_heart_raw = fs::path(outputDir) / "03_heart_raw_mask.png";
    cv::imwrite(savePath_heart_raw.string(), heartMask);
    fs::path savePath_bone_raw = fs::path(outputDir) / "04_bone_raw_mask.png";
    cv::imwrite(savePath_bone_raw.string(), boneMask);
    std::cout << "Guardado máscaras raw con rangos ajustados." << std::endl;


    // =========================================================
    // PASO 3: REFINAMIENTO MORFOLÓGICO Y FILTRADO DE BORDES
    // =========================================================
    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(5,5));
    
    // 3a. Morfología
    cv::morphologyEx(lungMask,  lungMask,  cv::MORPH_OPEN, kernel);
    cv::morphologyEx(heartMask, heartMask, cv::MORPH_OPEN, kernel);
    cv::morphologyEx(boneMask,  boneMask,  cv::MORPH_CLOSE, kernel);

    // 3b. Filtrar bordes (Aplicamos la inversa de los bordes dilatados)
    cv::Mat invEdges;
    cv::bitwise_not(edgesDilated, invEdges);

    // >>>>>>>>>>>>>>>>>>>>>>>>>>>>>> CORRECCIÓN AQUÍ <<<<<<<<<<<<<<<<<<<<<<<<<<<<
    // Usamos la sobrecarga de cv::bitwise_and que tiene 3 argumentos (src1, src2, dst)
    cv::bitwise_and(lungMask, invEdges, lungMask);
    cv::bitwise_and(heartMask, invEdges, heartMask);
    cv::bitwise_and(boneMask, invEdges, boneMask);
    // >>>>>>>>>>>>>>>>>>>>>>>>>>>>>> FIN CORRECCIÓN <<<<<<<<<<<<<<<<<<<<<<<<<<<<<

    fs::path savePath_lung_ref = fs::path(outputDir) / "05_lung_refined_mask.png";
    cv::imwrite(savePath_lung_ref.string(), lungMask);
    fs::path savePath_heart_ref = fs::path(outputDir) / "06_heart_refined_mask.png";
    cv::imwrite(savePath_heart_ref.string(), heartMask);
    fs::path savePath_bone_ref = fs::path(outputDir) / "07_bone_refined_mask.png";
    cv::imwrite(savePath_bone_ref.string(), boneMask);
    std::cout << "Guardado máscaras refinadas." << std::endl;
}

/**
 * Implementación del resaltado de color y guardado de la imagen final
 */
void MedicalImageProcessor::highlightAndSaveFinalImage(const cv::Mat& lungMask, const cv::Mat& heartMask, const cv::Mat& boneMask) {
    cv::Mat finalColorImage;
    cv::cvtColor(cvSlice8bit, finalColorImage, cv::COLOR_GRAY2BGR); // Base en color

    // Pulmón (Cian: B=255, G=255, R=0)
    finalColorImage.setTo(cv::Scalar(255, 255, 0), lungMask);
    // Corazón/Tejido blando (Rojo: B=0, G=0, R=255)
    finalColorImage.setTo(cv::Scalar(0, 0, 255), heartMask);
    // Hueso (Verde: B=0, G=255, R=0)
    finalColorImage.setTo(cv::Scalar(0, 255, 0), boneMask);

    fs::path savePath_final = fs::path(outputDir) / "12_final_highlighted.png";
    cv::imwrite(savePath_final.string(), finalColorImage);
    std::cout << "Guardado: " << savePath_final.string() << std::endl;
}


/**
 * @brief Ejecuta la secuencia final de procesamiento (Canny, Morfología, Guardado de Evidencia) 
 * usando los rangos HU ajustados por la GUI.
 */
bool MedicalImageProcessor::executeFinalProcessing() {
    if (cvSliceHU.empty()) {
        std::cerr << "Error: La imagen HU está vacía. No se puede ejecutar el procesamiento final." << std::endl;
        return false;
    }
    
    cv::Mat lungMask, heartMask, boneMask;
    try {
        // Ejecutar toda la cadena de procesamiento con los rangos HU actuales
        applyProcessingChain(lungMask, heartMask, boneMask);
        
        // Guardar la imagen final
        highlightAndSaveFinalImage(lungMask, heartMask, boneMask);

    } catch (const cv::Exception& e) {
        std::cerr << "Error OpenCV durante el procesamiento final: " << e.what() << std::endl;
        return false;
    }

    std::cout << "Proceso final completado. Evidencia guardada en: " << outputDir << std::endl;
    
    return true;
}

// NO SE INCLUYE main() - El punto de entrada está en main.cpp para la aplicación Qt