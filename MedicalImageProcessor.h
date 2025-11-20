#include <iostream>
#include <string>
#include <map>
#include <filesystem>
#include <itkImage.h>
#include <itkImageFileReader.h>
#include <itkImageRegionConstIterator.h>
#include <opencv2/opencv.hpp>

namespace fs = std::filesystem;

class MedicalImageProcessor {
public:
    // Estructura para definir los rangos de Unidades Hounsfield (HU)
    struct HURanges {
        short min;
        short max;
    };

    // ==========================================================
    // ACCESO PÚBLICO PARA LA GUI (Necesario para el control interactivo)
    // ==========================================================
    // Rangos HU por defecto para Pulmón, Corazón y Hueso (Mutable para ser ajustado por la GUI)
    std::map<std::string, HURanges> HU_RANGES = {
        {"lung",        {-1000, -300}},
        {"soft_tissue", {-100,  300}},
        {"bone",        {300,   3000}}
    };

    // Propiedades de configuración
    const short huOffset = 2000;
    
    // Getters para exponer las matrices a la GUI
    const cv::Mat& getSliceHU() const { return cvSliceHU; }
    const cv::Mat& getSlice8bit() const { return cvSlice8bit; }
    
    // <<<< NUEVA FUNCIÓN AÑADIDA >>>>
    const std::string& getOutputDir() const { return outputDir; } 
    // <<<< FIN DE NUEVA FUNCIÓN >>>>

private:
    std::string inputFilePath;
    std::string outputDir;
    
    // El slice de la imagen original en HU (16 bits)
    cv::Mat cvSliceHU;
    // El slice normalizado para visualización (8 bits)
    cv::Mat cvSlice8bit;
    
    /**
     * @brief Aplica la cadena de procesamiento (Umbralización, Canny, Morfología)
     * y guarda las imágenes intermedias, usando los rangos HU ajustados.
     * @param lungMask Máscara de Pulmón generada.
     * @param heartMask Máscara de Corazón/Tejido Blando generada.
     * @param boneMask Máscara de Hueso generada.
     */
    void applyProcessingChain(cv::Mat& lungMask, cv::Mat& heartMask, cv::Mat& boneMask);

    /**
     * @brief Genera la imagen final a color con las ROIs resaltadas y la guarda.
     * @param lungMask Máscara final de Pulmón.
     * @param heartMask Máscara final de Corazón.
     * @param boneMask Máscara final de Hueso.
     */
    void highlightAndSaveFinalImage(const cv::Mat& lungMask, const cv::Mat& heartMask, const cv::Mat& boneMask);

public:
    MedicalImageProcessor(const std::string& filePath, const std::string& outputDirectory);

    /**
     * @brief [PARTE I: FASE 1] Carga el archivo y prepara las matrices de trabajo.
     * (Esta es la única declaración de esta función.)
     * @param filePath Ruta al archivo.
     * @return true si la carga y conversión fueron exitosas.
     */
    bool loadAndConvertSlice(const std::string& filePath);

    /**
     * @brief [PARTE I: FASE 2] Ejecuta la secuencia completa de procesamiento y guardado (Canny, Morfología, Guardado) 
     * usando los rangos HU actualmente guardados en HU_RANGES (ajustados por la GUI).
     * @return true si el proceso fue exitoso.
     */
    bool executeFinalProcessing();
};