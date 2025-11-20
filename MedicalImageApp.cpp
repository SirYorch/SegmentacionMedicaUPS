#include "MedicalImageApp.h"
#include <QHBoxLayout>
#include <QVBoxLayout>
#include <QGridLayout> // Usado para el layout de los controles de rango
#include <QGroupBox>
#include <QIntValidator>
#include <QScrollArea> // Para hacer la zona de controles scrollable
#include <QFileInfo> // Para obtener el path de la carpeta de salida

// ====================================================================
// FUNCIÓN DE CONVERSIÓN: OpenCV (cv::Mat) a Qt (QImage)
// ====================================================================
QImage cvMatToQImage(const cv::Mat &mat) {
// ... (código de cvMatToQImage sin cambios)
    if (mat.empty()) {
        return QImage();
    }
    
    // Necesario para que Qt pueda manipular la memoria de OpenCV
    if (mat.type() == CV_8UC1) {
        // Escala de grises (8 bits, 1 canal)
        QImage image(mat.data, mat.cols, mat.rows, mat.step, QImage::Format_Grayscale8);
        return image.copy();
    } else if (mat.type() == CV_8UC3) {
        // BGR a RGB, necesario porque Qt espera RGB y OpenCV usa BGR por defecto
        cv::Mat rgb;
        cv::cvtColor(mat, rgb, cv::COLOR_BGR2RGB);
        QImage image(rgb.data, rgb.cols, rgb.rows, rgb.step, QImage::Format_RGB888);
        return image.copy();
    }
    return QImage();
}


// ====================================================================
// CONSTRUCTOR E INICIALIZACIÓN DE LA GUI
// ====================================================================

MedicalImageApp::MedicalImageApp(QWidget *parent)
    : QMainWindow(parent) {
    
    setWindowTitle("Visor Médico Interactivo (C++ Qt) | Parte I");
    resize(1200, 800);

    // 1. Área Central y Layouts
    QWidget *centralWidget = new QWidget(this);
    setCentralWidget(centralWidget);

    QHBoxLayout *mainLayout = new QHBoxLayout(centralWidget);
    
    // 2. Controles (Lado Izquierdo) - Contenido Scrollable
    QWidget *controlsContainer = new QWidget();
    controlsContainer->setFixedWidth(350);
    
    QScrollArea *scrollArea = new QScrollArea();
    scrollArea->setWidgetResizable(true);
    scrollArea->setWidget(controlsContainer);

    QVBoxLayout *controlsLayout = new QVBoxLayout(controlsContainer);
    
    selectFileButton = new QPushButton("1. Seleccionar Archivo (.ima, .nii, .dcm)");
    selectFileButton->setStyleSheet("font-weight: bold; padding: 10px; background-color: #4CAF50; color: white; border-radius: 8px;");
    connect(selectFileButton, &QPushButton::clicked, this, &MedicalImageApp::selectFile);
    controlsLayout->addWidget(selectFileButton);
    
    originalDisplayLabel = new QLabel("IMAGEN BASE CARGADA");
    originalDisplayLabel->setAlignment(Qt::AlignCenter);
    originalDisplayLabel->setFixedSize(300, 300);
    originalDisplayLabel->setFrameStyle(QFrame::Box | QFrame::Sunken);
    originalDisplayLabel->setStyleSheet("background-color: #333; color: #FFF; border-radius: 8px;");
    controlsLayout->addWidget(originalDisplayLabel);
    
    // Controles de ROI
    QGroupBox *roiBox = new QGroupBox("2. Ajuste Dinámico de Rangos HU (Segmentación)");
    roiBox->setStyleSheet("QGroupBox { font-weight: bold; margin-top: 10px; border: 1px solid gray; border-radius: 8px; padding-top: 20px;}");
    QVBoxLayout *roiLayout = new QVBoxLayout(roiBox);

    // ========================================================================
    // CORRECCIÓN: Usar una ruta válida para la instancia temporal del procesador
    // Esto previene el error 'filesystem error: cannot create directories: Invalid argument'
    // ========================================================================
    MedicalImageProcessor tempProcessor("", "temp_output_dir"); 
    
    roiLayout->addWidget(createRangeControl("Pulmón (Cian) - Aire", lungMinSlider, lungMaxSlider, lungMinEdit, lungMaxEdit, tempProcessor.HU_RANGES.at("lung").min, tempProcessor.HU_RANGES.at("lung").max));
    roiLayout->addWidget(createRangeControl("Tejido Blando/Corazón (Rojo)", softMinSlider, softMaxSlider, softMinEdit, softMaxEdit, tempProcessor.HU_RANGES.at("soft_tissue").min, tempProcessor.HU_RANGES.at("soft_tissue").max));
    roiLayout->addWidget(createRangeControl("Hueso (Verde) - Denso", boneMinSlider, boneMaxSlider, boneMinEdit, boneMaxEdit, tempProcessor.HU_RANGES.at("bone").min, tempProcessor.HU_RANGES.at("bone").max));

    controlsLayout->addWidget(roiBox);
    
    saveButton = new QPushButton("3. Finalizar, Aplicar Morfología y Guardar Evidencia");
    saveButton->setStyleSheet("font-weight: bold; padding: 10px; background-color: #008CBA; color: white; border-radius: 8px;");
    saveButton->setEnabled(false);
    connect(saveButton, &QPushButton::clicked, this, &MedicalImageApp::finalizeAndSave);
    controlsLayout->addWidget(saveButton);
    
    controlsLayout->addStretch(1);
    mainLayout->addWidget(scrollArea);

    // 3. Display de la Imagen Procesada (Lado Derecho)
    imageDisplayLabel = new QLabel("Seleccione un archivo para comenzar...");
    imageDisplayLabel->setAlignment(Qt::AlignCenter);
    imageDisplayLabel->setMinimumSize(400, 400);
    imageDisplayLabel->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Expanding);
    imageDisplayLabel->setFrameStyle(QFrame::Box | QFrame::Raised);
    imageDisplayLabel->setStyleSheet("background-color: #EEE; font-size: 16px; border-radius: 8px;");
    mainLayout->addWidget(imageDisplayLabel);
}

MedicalImageApp::~MedicalImageApp() {}

// ====================================================================
// FUNCIÓN AUXILIAR PARA CREAR CONTROLES
// ====================================================================

QWidget* MedicalImageApp::createRangeControl(const QString& title, QSlider*& minSlider, QSlider*& maxSlider, 
                                            QLineEdit*& minEdit, QLineEdit*& maxEdit, short defaultMin, short defaultMax) {
    
    QGroupBox *group = new QGroupBox(title);
    group->setStyleSheet("QGroupBox { border: 1px solid #CCC; border-radius: 4px; padding-top: 15px;}");
    QGridLayout *layout = new QGridLayout(group); 
    
    // Slider configuration
    minSlider = new QSlider(Qt::Horizontal);
    maxSlider = new QSlider(Qt::Horizontal);
    
    // Configuración general del slider (offset para manejar números negativos)
    minSlider->setRange(0, MAX_SLIDER_VALUE);
    maxSlider->setRange(0, MAX_SLIDER_VALUE);
    
    // Valores iniciales (HU + offset)
    minSlider->setValue(defaultMin + HU_OFFSET);
    maxSlider->setValue(defaultMax + HU_OFFSET);

    // LineEdit configuration
    minEdit = new QLineEdit(QString::number(defaultMin));
    maxEdit = new QLineEdit(QString::number(defaultMax));
    minEdit->setFixedWidth(60);
    maxEdit->setFixedWidth(60);
    
    // Usar un validador para permitir solo números enteros (aproximadamente -4000 a 4000)
    QIntValidator *validator = new QIntValidator(-4000, 4000, this);
    minEdit->setValidator(validator);
    maxEdit->setValidator(validator);

    // Layout
    // Fila para Min
    layout->addWidget(new QLabel("Min HU:"), 0, 0);
    layout->addWidget(minSlider, 0, 1);
    layout->addWidget(minEdit, 0, 2);

    // Fila para Max
    layout->addWidget(new QLabel("Max HU:"), 1, 0);
    layout->addWidget(maxSlider, 1, 1);
    layout->addWidget(maxEdit, 1, 2);

    // Conectar eventos de sincronización (Slider <-> LineEdit)
    connectSliderToLineEdit(minSlider, minEdit);
    connectSliderToLineEdit(maxSlider, maxEdit);
    
    // Conectar eventos de cambio a la función de actualización de imagen (Dinámica)
    connect(minSlider, &QSlider::valueChanged, this, &MedicalImageApp::updateHURanges);
    connect(maxSlider, &QSlider::valueChanged, this, &MedicalImageApp::updateHURanges);
    connect(minEdit, &QLineEdit::editingFinished, this, &MedicalImageApp::updateHURanges);
    connect(maxEdit, &QLineEdit::editingFinished, this, &MedicalImageApp::updateHURanges);
    
    return group;
}

// ====================================================================
// CONEXIONES Y SLOTS
// ====================================================================

void MedicalImageApp::connectSliderToLineEdit(QSlider* slider, QLineEdit* lineEdit) {
    // 1. Slider -> LineEdit: Convierte valor (con offset) a HU real
    connect(slider, &QSlider::valueChanged, [this, lineEdit](int value) {
        // Solo actualiza el texto si no está en medio de una edición para evitar loops
        if (!lineEdit->hasFocus()) {
            lineEdit->setText(QString::number(value - HU_OFFSET));
        }
    });

    // 2. LineEdit -> Slider: Convierte HU real a valor (con offset)
    connect(lineEdit, &QLineEdit::editingFinished, [this, slider, lineEdit]() {
        bool ok;
        int value = lineEdit->text().toInt(&ok);
        if (ok) {
            slider->setValue(value + HU_OFFSET);
        }
    });
}

void MedicalImageApp::selectFile() {
    // CORRECCIÓN CLAVE: Usar un filtro de archivos más amplio para .ima y archivos sin extensión.
    // "Archivos Médicos (*.ima *.nii *.nii.gz *.dcm);;Todos los archivos (*.*)"
    QString filter = "Archivos Médicos (*.ima *.nii *.nii.gz *.dcm);;Todos los archivos (*)";
    
    QString filePath = QFileDialog::getOpenFileName(this, "Seleccionar Archivo Médico", QDir::homePath(), filter);

    if (!filePath.isEmpty()) {
        QString outputBaseName = QFileInfo(filePath).baseName();
        // Construye el nombre de la carpeta de salida basada en el nombre del archivo
        QString outputDir = "output_" + outputBaseName;
        
        // Inicializar el procesador con la ruta
        processor = std::make_unique<MedicalImageProcessor>(filePath.toStdString(), outputDir.toStdString());
        
        // Cargar el archivo y obtener las matrices base
        if (processor->loadAndConvertSlice(filePath.toStdString())) { 
            
            QImage originalImage = cvMatToQImage(processor->getSlice8bit());
            if (!originalImage.isNull()) {
                // Mostrar imagen original
                originalDisplayLabel->setPixmap(QPixmap::fromImage(originalImage).scaled(originalDisplayLabel->size(), Qt::KeepAspectRatio, Qt::SmoothTransformation));
                saveButton->setEnabled(true);
                imageDisplayLabel->setText("Corte cargado. Ajuste los rangos HU.");
                
                // Disparar el primer procesamiento dinámico con los valores por defecto
                updateHURanges(); 
            } else {
                QMessageBox::critical(this, "Error de Imagen", "No se pudo obtener la imagen base 8-bit del procesador.");
            }
        } else {
            QMessageBox::critical(this, "Error de Carga", "No se pudo cargar el archivo médico. Verifique la ruta y el formato (ITK/DICOM).");
            saveButton->setEnabled(false);
        }
    }
}

void MedicalImageApp::updateImageDisplay() {
    // Si el procesador no está listo o no hay datos, salir
    if (!processor || processor->getSliceHU().empty()) return;

    // 1. Obtener los rangos HU ajustados de la GUI
    short lungMin = lungMinEdit->text().toShort();
    short lungMax = lungMaxEdit->text().toShort();
    short softMin = softMinEdit->text().toShort();
    short softMax = softMaxEdit->text().toShort();
    short boneMin = boneMinEdit->text().toShort();
    short boneMax = boneMaxEdit->text().toShort();

    // 2. Obtener las matrices base del procesador (HU y 8-bit)
    const cv::Mat& currentSliceHU = processor->getSliceHU();
    const cv::Mat& currentSlice8bit = processor->getSlice8bit();

    cv::Mat lungMask, softMask, boneMask;
    
    // Umbralización (se hace aquí para la visualización dinámica)
    cv::inRange(currentSliceHU, cv::Scalar(lungMin), cv::Scalar(lungMax), lungMask);
    cv::inRange(currentSliceHU, cv::Scalar(softMin), cv::Scalar(softMax), softMask);
    cv::inRange(currentSliceHU, cv::Scalar(boneMin), cv::Scalar(boneMax), boneMask);

    cv::Mat highlightedImage;
    cv::cvtColor(currentSlice8bit, highlightedImage, cv::COLOR_GRAY2BGR);

    // 3. Pintar Máscaras (BGR)
    highlightedImage.setTo(cv::Scalar(255, 255, 0), lungMask);  // Pulmón: Cian
    highlightedImage.setTo(cv::Scalar(0, 0, 255), softMask);   // Corazón: Rojo
    highlightedImage.setTo(cv::Scalar(0, 255, 0), boneMask);   // Hueso: Verde

    // 4. Mostrar en Qt
    QImage qimg = cvMatToQImage(highlightedImage);
    if (!qimg.isNull()) {
        imageDisplayLabel->setPixmap(QPixmap::fromImage(qimg).scaled(imageDisplayLabel->size(), Qt::KeepAspectRatio, Qt::SmoothTransformation));
    }
}

void MedicalImageApp::updateHURanges() {
    // El slot principal para la actualización del display cuando se mueve un control
    updateImageDisplay();
}

void MedicalImageApp::finalizeAndSave() {
    if (!processor || processor->getSliceHU().empty()) {
        QMessageBox::warning(this, "Advertencia", "Debe cargar una imagen primero.");
        return;
    }
    
    // 1. Actualizar el mapa interno del procesador con los rangos ajustados por la GUI
    short lungMin = lungMinEdit->text().toShort();
    short lungMax = lungMaxEdit->text().toShort();
    short softMin = softMinEdit->text().toShort();
    short softMax = softMaxEdit->text().toShort();
    short boneMin = boneMinEdit->text().toShort();
    short boneMax = boneMaxEdit->text().toShort();

    // Sincronizar los rangos en el procesador antes de llamar a executeFinalProcessing
    processor->HU_RANGES["lung"].min = lungMin;
    processor->HU_RANGES["lung"].max = lungMax;
    processor->HU_RANGES["soft_tissue"].min = softMin;
    processor->HU_RANGES["soft_tissue"].max = softMax;
    processor->HU_RANGES["bone"].min = boneMin;
    processor->HU_RANGES["bone"].max = boneMax;
    
    // Deshabilitar botón mientras se procesa la cadena larga
    saveButton->setEnabled(false);
    
    // 2. Ejecutar la secuencia final de Canny, Morfología y Guardado de Evidencia
    if (processor->executeFinalProcessing()) { 
         // Usamos el getter getOutputDir()
         QString outputAbsolutePath = QFileInfo(QString::fromStdString(processor->getOutputDir())).absoluteFilePath();
         QMessageBox::information(this, "Éxito", "El procesamiento final (Canny, Morfología) y toda la evidencia (01_ a 07_) han sido guardados en la carpeta: " + outputAbsolutePath);
    } else {
         QMessageBox::critical(this, "Error", "Fallo al ejecutar el procesamiento final.");
    }
    
    saveButton->setEnabled(true);
}