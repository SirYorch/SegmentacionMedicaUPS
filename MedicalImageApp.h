#include <QMainWindow>
#include <QLabel>
#include <QSlider>
#include <QLineEdit>
#include <QImage>
#include <QPushButton>
#include <QFileDialog>
#include <QMessageBox>
#include <QDebug> // Para logs

#include "MedicalImageProcessor.h"
#include <opencv2/opencv.hpp>

// Clase auxiliar para convertir cv::Mat a QImage
QImage cvMatToQImage(const cv::Mat &mat);

class MedicalImageApp : public QMainWindow {
    Q_OBJECT // Macro esencial para el sistema de señales y slots de Qt

public:
    MedicalImageApp(QWidget *parent = nullptr);
    ~MedicalImageApp();

private slots:
    // Slots para la interacción del usuario
    void selectFile();
    void updateHURanges(); // Se llama cuando cualquier slider se mueve
    void finalizeAndSave(); // Se llama al presionar el botón de guardar

private:
    // Widgets de la interfaz
    QLabel *imageDisplayLabel; // Para mostrar la imagen procesada
    QLabel *originalDisplayLabel; // Para mostrar la imagen base
    
    // Controles de Umbralización (Sliders y Cajas de Texto)
    QSlider *lungMinSlider, *lungMaxSlider;
    QSlider *softMinSlider, *softMaxSlider;
    QSlider *boneMinSlider, *boneMaxSlider;
    
    QLineEdit *lungMinEdit, *lungMaxEdit;
    QLineEdit *softMinEdit, *softMaxEdit;
    QLineEdit *boneMinEdit, *boneMaxEdit;
    
    QPushButton *selectFileButton;
    QPushButton *saveButton;

    // Lógica del Procesamiento (Parte I)
    std::unique_ptr<MedicalImageProcessor> processor;
    
    // Estado de la imagen
    cv::Mat currentSliceHU; // La matriz de HU cargada
    cv::Mat currentSlice8bit; // La matriz normalizada 8-bit

    // Propiedades de configuración
    const short HU_OFFSET = 2000;
    const int MAX_SLIDER_VALUE = 4096 + 2000; // Rango total de HU + offset

    /**
     * @brief Construye los sliders y campos de texto para un grupo (ej. Pulmón).
     */
    QWidget* createRangeControl(const QString& title, QSlider*& minSlider, QSlider*& maxSlider, 
                                QLineEdit*& minEdit, QLineEdit*& maxEdit, short defaultMin, short defaultMax);
    
    /**
     * @brief Aplica la umbralización HU a la imagen base y actualiza el display.
     */
    void updateImageDisplay();
    
    /**
     * @brief Sincroniza el valor del slider al valor del campo de texto (y viceversa).
     */
    void connectSliderToLineEdit(QSlider* slider, QLineEdit* lineEdit);
};