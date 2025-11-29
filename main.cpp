#include <iostream>
#include <vector>
#include <string>
#include <algorithm>
#include <filesystem>
#include <opencv2/opencv.hpp>
#include <itkImage.h>
#include <itkImageFileReader.h>
#include <itkRescaleIntensityImageFilter.h>
#include <itkGDCMImageIO.h>

using namespace std;
using namespace cv;
using namespace itk;

// --- NUEVO: VARIABLES GLOBALES PARA EL MENÚ ---
Mat img_menu_normal;
// Mat img_menu_click;
bool enMenu = true; // Indica si estamos en el menú o en la app

// ¡¡¡¡ IMPORTANTE: CALIBRA ESTOS VALORES !!!!
// Usa los couts de la consola para ver donde haces clic y ajustar esto
const int BTN_INICIAR_X1 = 100, BTN_INICIAR_X2 = 300;
const int BTN_INICIAR_Y1 = 200, BTN_INICIAR_Y2 = 250;
// ----------------------------------------------

using InputPixelType  = short;
using OutputPixelType = unsigned char;
using InputImageType  = Image<InputPixelType, 2>;
using OutputImageType = Image<OutputPixelType, 2>;

int currentSlice = 0;
int ksizeTrack = 3;       
int sigmaTrack = 10;      
int umbralMin = 0;
int umbralMax = 255;
int cx = 100;
int cy = 100;
int radioX = 50;
int radioY = 50;

int kernelSize = 3;

bool clahe = false;
bool eq = false;

// Declaración global de botones de la APP (no del menú)
Rect btnCLAHE(10, 10, 120, 40);
Rect btnEq(10, 60, 120, 40);

// ... [TUS FUNCIONES AUXILIARES: ITKToMat, readRaw, readIMA, etc. SE MANTIENEN IGUAL] ...
// (Las pego colapsadas para ahorrar espacio, el código lógico no cambia)

void updateKSize(int KSize){ KSize = KSize*2+1; }

Mat Umbrilize(Mat imagen, int umbralMin, int umbralMax){
    Mat img = Mat::zeros(imagen.rows, imagen.cols, CV_8UC1);
    for(int i =0 ; i < imagen.rows ; i ++){
        for(int j =0 ; j < imagen.cols ; j ++){
            uchar pixel = imagen.at<uchar>(i, j);
            if (pixel >= umbralMin && pixel <= umbralMax) img.at<uchar>(i, j) = 255;     
            else img.at<uchar>(i, j) = 0;       
        }   
    }
    return img;
}

Mat ITKToMat(OutputImageType::Pointer img) {
    auto size = img->GetLargestPossibleRegion().GetSize();
    Mat out(size[1], size[0], CV_8UC1);
    memcpy(out.data, img->GetBufferPointer(), size[0] * size[1]);
    return out;
}

Mat readIMA(const string& filename) {
    try {
        using ImageIOType = GDCMImageIO;
        ImageIOType::Pointer dicomIO = ImageIOType::New();
        using ReaderType = ImageFileReader<InputImageType>;
        ReaderType::Pointer reader = ReaderType::New();
        reader->SetImageIO(dicomIO);
        reader->SetFileName(filename);
        reader->Update();
        using RescaleType = RescaleIntensityImageFilter<InputImageType, OutputImageType>;
        RescaleType::Pointer scale = RescaleType::New();
        scale->SetInput(reader->GetOutput());
        scale->SetOutputMinimum(0);
        scale->SetOutputMaximum(255);
        scale->Update();
        return ITKToMat(scale->GetOutput());
    }
    catch (...) {
        cout << "Error leyendo como IMA/DICOM: " << filename << "\n";
        return Mat();
    }
}

Mat toClahe(Mat imagen){
    Mat salida;
    Ptr<CLAHE> clahe = createCLAHE(2.0, cv::Size(8, 8));
    clahe->apply(imagen, salida);
    return salida;
}

Mat toEq(Mat imagen){
    equalizeHist(imagen, imagen);
    return imagen;
}

Mat toGaussianBlur(Mat imagen, int kernelSize = 5, double sigma = 0) {
    Mat salida;
    if (kernelSize % 2 == 0) kernelSize++;
    GaussianBlur(imagen, salida, cv::Size(kernelSize, kernelSize), sigma);
    return salida;
}

vector<string> getIMA(const string& dir) {
    vector<string> files;
    for (auto& e : filesystem::directory_iterator(dir)) {
        if (!e.is_regular_file()) continue;
        string ext = e.path().extension().string();
        transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
        if (ext == ".ima") files.push_back(e.path().string());
    }
    sort(files.begin(), files.end());
    return files;
}

Mat maskCircle(Mat &img, int cx, int cy, int rx, int ry) {
    Mat mask = Mat::zeros(img.rows, img.cols, CV_8UC1);
    ellipse(mask, cv::Point(cx, cy), cv::Size(rx, ry), 0, 0, 360, Scalar(255), FILLED);
    Mat salida(img.rows, img.cols, img.type(), Scalar(255));
    img.copyTo(salida, mask);
    return salida;
}

Mat open(Mat imagen, int kernelSize){
    Mat erodida, salida;
    Mat kernel = getStructuringElement(MORPH_RECT, cv::Size(kernelSize, kernelSize));
    erode(imagen, erodida, kernel);
    dilate(erodida, salida, kernel);
    return salida;
}

Mat close(Mat imagen, int kernelSize){
    Mat dilatada, salida;
    Mat kernel = getStructuringElement(MORPH_RECT, cv::Size(kernelSize, kernelSize));
    dilate(imagen, dilatada, kernel);
    erode(dilatada, salida, kernel);
    return salida;
}

Mat boneWindowing(Mat imagen, int minVal, int maxVal) {
    Mat salida = imagen.clone();
    double escala = 255.0 / (double)(maxVal - minVal);
    imagen.convertTo(salida, -1, escala, -minVal * escala);
    return salida;
}

// ... [INCLUIR AQUÍ TUS FUNCIONES defineBones, defineOrgan, mergeMasks, etc. INTACTAS] ...
// (Para que el código compile, asumo que están definidas tal cual me las pasaste arriba)
// POR BREVEDAD, SOLO PONGO LOS PROTOTIPOS AQUÍ, PERO EN TU CÓDIGO DÉJALAS COMPLETAS:
Mat defineBones(Mat imagen); 
Mat defineOrgan(Mat imagen, bool isLung);
Mat mergeMasks(Mat &imgOriginal, Mat &maskLung, float alpha = 0.3);
Mat mergeMasksHeart(Mat &imgOriginal, Mat &maskLung, float alpha = 0.3);
Mat mergeMasksBones(Mat &imgOriginal, Mat &maskBone, float alpha = 0.3);
// Mat mejorarNitidez(Mat imagenEntrada);


struct Button {
    int x1, y1, x2, y2;
};
 
Button btn_corazon  = {50, 225, 147, 252};
Button btn_huesos   = {200, 225, 300, 252};
Button btn_pulmones = {359, 225, 460, 252};

Button btn_result   = {50, 297, 147, 330};
Button btn_comp     = {200, 297, 300, 330};
Button btn_extra    = {359, 297, 460, 330};

Button btn_pruebas    = {200, 345, 300, 375};

bool inside(Button b, int x, int y) {
    return (x > b.x1 && x < b.x2 && y > b.y1 && y < b.y2);
}

void drawMask(Mat &img, Button b) {
    Rect rect(b.x1, b.y1, b.x2 - b.x1, b.y2 - b.y1);

    // Crear una copia de la imagen original para el overlay
    Mat overlay;
    img.copyTo(overlay);

    // Color del rectángulo (BGR)
    Scalar color(255, 100, 100); // Azul suave
    double alpha = 0.2; // 20% de opacidad

    // Dibujar el rectángulo lleno en el overlay
    rectangle(overlay, rect, color, FILLED);

    // Mezclar overlay + original según la máscara del rectángulo
    addWeighted(overlay, alpha, img, 1 - alpha, 0, img);
}

Mat img_normal;
Mat img_click; 

// --- MODIFICACIÓN CLAVE: MOUSE CALLBACK UNIFICADO ---
void onMouse(int event, int x, int y, int flags, void* userdata) {
    if (event == EVENT_LBUTTONDOWN) {

        cout << "Click: " << x << " " << y << endl;

        // Copiamos la imagen normal
        img_click = img_normal.clone();

        // ---- BOTON CORAZÓN ----
        if (inside(btn_corazon, x, y)) {
            drawMask(img_click, btn_corazon);
            imshow("Aplicacion Principal", img_click);
            waitKey(120);   // pequeña pausa visual
            cout << "Botón: CORAZÓN\n";
        }

        // ---- BOTON HUESOS ----
        else if (inside(btn_huesos, x, y)) {
            drawMask(img_click, btn_huesos);
            imshow("Aplicacion Principal", img_click);
            waitKey(120);
            cout << "Botón: HUESOS\n";
        }

        // ---- BOTON PULMONES ----
        else if (inside(btn_pulmones, x, y)) {
            drawMask(img_click, btn_pulmones);
            imshow("Aplicacion Principal", img_click);
            waitKey(120);
            cout << "Botón: PULMONES\n";
        }

        // ---- RESULTADOS ----
        else if (inside(btn_result, x, y)) {
            drawMask(img_click, btn_result);
            imshow("Aplicacion Principal", img_click);
            waitKey(120);
            cout << "Botón: RESULTADOS\n";
        }

        // ---- COMPARACION DnCNN ----
        else if (inside(btn_comp, x, y)) {
            drawMask(img_click, btn_comp);
            imshow("Aplicacion Principal", img_click);
            waitKey(120);
            cout << "Botón: COMPARACION DnCNN\n";
        }

        // ---- EXTRA ----
        else if (inside(btn_extra, x, y)) {
            drawMask(img_click, btn_extra);
            imshow("Aplicacion Principal", img_click);
            waitKey(120);
            cout << "Botón: EXTRA\n";
        }

        // ---- PRUEBAS ----
        
        else if (inside(btn_pruebas, x, y)) {
            drawMask(img_click, btn_pruebas);
            imshow("Aplicacion Principal", img_click);
            waitKey(120);
            cout << "Botón: PRUEBAS\n";
        }

        // Restauramos la imagen normal después de la animación
        imshow("Aplicacion Principal", img_normal);
    }
}

// ---------------- MAIN ----------------
int main() {
    // 1. CARGA DE RECURSOS (Imágenes DICOM y Menú)
    
    // Cargar Menu
    img_normal = imread("img_normal.png");
    // img_menu_click = imread("img_click.png");

    resize(img_normal,img_normal,cv::Size(500, 395));

    // if (img_menu_normal.empty() || img_menu_click.empty()) {
    //     cout << "ADVERTENCIA: No se encontraron menu_normal.png o menu_click.png. Creando fondo negro." << endl;
    //     img_menu_normal = Mat::zeros(600, 800, CV_8UC3);
    //     img_menu_click = Mat::zeros(600, 800, CV_8UC3);
     
    //     putText(img_menu_normal, "MENU (Faltan imagenes)", cv::Point(50,300), FONT_HERSHEY_SIMPLEX, 1, Scalar(255,255,255));
    // }

    // Cargar DICOMS (ITK)
    string folder = "L333";
    auto files = getIMA(folder);
    vector<Mat> imgs;
    imgs.reserve(files.size());

    cout << "Cargando imagenes DICOM/IMA..." << endl;
    for (auto& f : files) {
        Mat m = readIMA(f);
        if (m.empty()) m = Mat::zeros(256, 256, CV_8UC1);
        imgs.push_back(m);
    }
    if (imgs.empty()) {
        cout << "No se pudo cargar ninguna imagen DICOM\n";
        return 1;
    }
    cout << "Carga completa." << endl;

    // 2. CONFIGURACIÓN DE VENTANA PRINCIPAL
    // Usaremos "Aplicacion Principal" para el Menú y luego para los Controles
    namedWindow("Aplicacion Principal", WINDOW_AUTOSIZE);
    setMouseCallback("Aplicacion Principal", onMouse);
    
    // Variables de control de ventanas creadas
    bool ventanasCreadas = false; 

    // 3. BUCLE DE EJECUCIÓN
    while (true) {
        
        // --- CASO A: ESTAMOS EN EL MENÚ ---
        if (enMenu) {
            imshow("Aplicacion Principal", img_normal);
            // Esperamos clic
            if (waitKey(30) == 27) break; // ESC para salir
        }
        
        // --- CASO B: ESTAMOS EN LA APP (VISUALIZACIÓN) ---
        else {
            
            // Inicialización única al entrar a la app (Crear ventanas)
            if (!ventanasCreadas) {
                // Reconfigurar la ventana "Aplicacion Principal" para que sea "Controles"
                // resizeWindow("Aplicacion Principal", 300, 100); 
                createTrackbar("Slice", "Aplicacion Principal", &currentSlice, imgs.size() - 1);
                
                // Crear el resto de ventanas
                namedWindow("Pulmones", WINDOW_AUTOSIZE);
                namedWindow("CompletaPulmones", WINDOW_AUTOSIZE);
                namedWindow("CorazonMascara", WINDOW_AUTOSIZE);
                namedWindow("CompletaCorazon", WINDOW_AUTOSIZE); 
                namedWindow("HuesosMascara", WINDOW_AUTOSIZE);
                namedWindow("CompletaHuesos", WINDOW_AUTOSIZE); 

                moveWindow("Pulmones", 300, 500);
                moveWindow("CompletaPulmones", 300, 0);
                moveWindow("CorazonMascara", 810, 500); 
                moveWindow("CompletaCorazon", 810, 0); 
                moveWindow("HuesosMascara", 1320, 500); 
                moveWindow("CompletaHuesos", 1320, 0); 

                ventanasCreadas = true;
                
                // Dibujar panel inicial de controles
                Mat panel(100, 300, CV_8UC3, Scalar(50, 50, 50));
                rectangle(panel, btnCLAHE, Scalar(200,200,200), FILLED);
                putText(panel, "CLAHE", cv::Point(25, 35), FONT_HERSHEY_SIMPLEX, 0.7, Scalar(0,0,0));
                rectangle(panel, btnEq, Scalar(200,200,200), FILLED);
                putText(panel, "EQ HIST", cv::Point(35, 85), FONT_HERSHEY_SIMPLEX, 0.7, Scalar(0,0,0));
                imshow("Aplicacion Principal", panel);
            }

            // --- LÓGICA DE PROCESAMIENTO EXISTENTE ---
            Mat Original = imgs[currentSlice].clone();
            if (clahe) Original = toClahe(Original); // Aplicar filtros si están activos
            if (eq) Original = toEq(Original);

            // Pulmones
            Mat pulmonesMascara = imgs[currentSlice].clone();
            pulmonesMascara = defineOrgan(pulmonesMascara, true); 
            Mat completaPulmones = mergeMasks(Original, pulmonesMascara, 0.3);
            // Mat imagenFinal = mejorarNitidez(completaPulmones);

            imshow("Pulmones", pulmonesMascara);
            imshow("CompletaPulmones", completaPulmones);
            // imshow("NitidaPulmones", imagenFinal); // Opcional

            // Corazon
            Mat corazonMascara = imgs[currentSlice].clone();
            corazonMascara = defineOrgan(corazonMascara, false);
            Mat completaCorazon = mergeMasksHeart(Original, corazonMascara, 0.3);

            imshow("CorazonMascara", corazonMascara);
            imshow("CompletaCorazon", completaCorazon);

            // Huesos
            Mat huesosMascara = imgs[currentSlice].clone();
            huesosMascara = defineBones(huesosMascara);
            Mat completaHuesos = mergeMasksBones(Original, huesosMascara, 0.4);

            imshow("HuesosMascara", huesosMascara);
            imshow("CompletaHuesos", completaHuesos);

            int k = waitKey(30);
            if (k == 27) break; // ESC para salir
        }
    }

    destroyAllWindows();
    return 0;
}

// COPIA AQUÍ ABAJO EL RESTO DE TUS FUNCIONES (defineOrgan, defineBones, etc.)
// ...

Mat defineBones(Mat imagen) {
    Mat procesada;
    procesada = boneWindowing(imagen, 120, 255);
    cv::GaussianBlur(imagen, procesada, cv::Size(3, 3), 0);
    procesada = toClahe(procesada);

    // 1. UMBRALIZACIÓN 
    int t_strong = 170; 
    int t_weak = 158; 

    Mat fuerte, debil;
    cv::threshold(procesada, fuerte, t_strong, 255, cv::THRESH_BINARY);
    cv::threshold(procesada, debil, t_weak, 255, cv::THRESH_BINARY);

    // -------------------------------------------------------------
   //EROSIÓN 
    // -------------------------------------------------------------
    Mat kernelErosion = cv::getStructuringElement(cv::MORPH_CROSS, cv::Size(3, 3));
    cv::erode(debil, debil, kernelErosion);
    // -------------------------------------------------------------

    Mat labels, stats, centroids;
    int nLabels = cv::connectedComponentsWithStats(debil, labels, stats, centroids);

    Mat mascaraBase = Mat::zeros(imagen.size(), CV_8UC1);

    // Filtro de área (INTACTO)
    int minArea = 25; 

    // --- PARÁMETROS DE LA ZONA DE MUERTE ---
    int zmX = 263;      
    int zmY = 210;      
    int zmRadio = 94;   
    int areaColumnaGigante = 10000; 

    // --- CONFIGURACIÓN DEL INTERVALO DE SLICES ---
    // La zona de muerte solo se activa en estos slices
    int inicioZonaMuerte = 245; 
    int finZonaMuerte = 438;
    // ---------------------------------------------

    for(int i = 1; i < nLabels; i++) {
        // Filtro básico de ruido
        if (stats.at<int>(i, cv::CC_STAT_AREA) < minArea) continue;

        Mat maskIsla = (labels == i);

        // Solo aplicamos esto si estamos en los slices del estómago
        if (currentSlice >= inicioZonaMuerte && currentSlice <= finZonaMuerte) {
            
            double cX = centroids.at<double>(i, 0);
            double cY = centroids.at<double>(i, 1);

            // Distancia euclidiana al centro de la zona de muerte
            double dist = std::sqrt(std::pow(cX - zmX, 2) + std::pow(cY - zmY, 2));

            // Si el objeto está DENTRO del círculo de muerte
            if (dist < zmRadio) {
                // Si NO es la columna gigante, lo matamos.
                if (stats.at<int>(i, cv::CC_STAT_AREA) < areaColumnaGigante) {
                    continue; // Está en la zona prohibida y es pequeño -> BORRAR
                }
            }
        }

        double minVal, maxVal;
        cv::minMaxLoc(imagen, &minVal, &maxVal, NULL, NULL, maskIsla);

        double umbralDureza = 180.0;

        if (maxVal < umbralDureza) {
            continue; // No tiene "corazón duro" -> BORRAR
        }
        // -------------------------------------------------------------

        cv::bitwise_or(mascaraBase, maskIsla, mascaraBase);
    }


    // --- RELLENO Y RESTAURACIÓN (INTACTO) ---
    Mat resultadoFinal = mascaraBase.clone();

    // Dilatación para recuperar lo erosionado
    cv::dilate(resultadoFinal, resultadoFinal, kernelErosion);

    // Cierre
    Mat kernelCierre = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(7, 7));
    cv::morphologyEx(resultadoFinal, resultadoFinal, cv::MORPH_CLOSE, kernelCierre);

    // Relleno final
    vector<vector<cv::Point>> contours;
    cv::findContours(resultadoFinal, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
    for(size_t i = 0; i < contours.size(); i++) {
        cv::drawContours(resultadoFinal, contours, (int)i, Scalar(255), cv::FILLED);
    }
    
    return resultadoFinal;
}

Mat defineOrgan(Mat imagen, bool isLung) {

    int cxdf, cydf, radioXdf, radioYdf;
    int mindf, maxdf, kerneldf;
    int sliceLimit;

    double sigmadf = sigmaTrack / 10.0;
    
    if (isLung) {
        // Parámetros para PULMONES (Umbral bajo, Apertura, Máscara grande)
        cxdf = 253;
        cydf = 264;
        radioXdf = 210;
        radioYdf = 147;
        mindf = 0;
        maxdf = 58;
        kerneldf = 9;
        sliceLimit = 127; 
        
    } else {

        cxdf = 284;
        cydf = 220;
        radioXdf = 87;
        radioYdf = 65;
        mindf = 110;  // Umbral ajustado para tejido blando
        maxdf = 170;
        kerneldf = 3; 
        sliceLimit = 40; 
    }


    if (currentSlice < sliceLimit) {
        imagen = maskCircle(imagen, cxdf, cydf, radioXdf, radioYdf);
        GaussianBlur(imagen, imagen, cv::Size(kerneldf, kerneldf), sigmadf);

        // Umbralización
        imagen = Umbrilize(imagen, mindf, maxdf);
        
        // Operación Morfológica Condicional
        if (isLung) {
            imagen = open(imagen, kernelSize); // Pulmones usan Apertura

        } else {
            imagen = open(imagen, kernelSize); // Corazón usa Cierre
            imagen = close(imagen, kernelSize); // Corazón usa Cierre

            vector<vector<cv::Point>> contours;
            vector<Vec4i> hierarchy;
            // Encontrar contornos externos
            findContours(imagen, contours, hierarchy, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

            double maxArea = 0;
            int maxAreaIdx = -1;
            // Buscar el índice del contorno con el área más grande
            for (size_t i = 0; i < contours.size(); i++) {
                double area = contourArea(contours[i]);
                if (area > maxArea) {
                    maxArea = area;
                    maxAreaIdx = i;
                }
            }
            
            Mat cleanMask = Mat::zeros(imagen.size(), CV_8UC1);
            
            if (maxAreaIdx >= 0) {
                drawContours(cleanMask, contours, maxAreaIdx, Scalar(255), FILLED);
            }
            
            imagen = cleanMask;

        }

    } else {
        imagen = Mat::zeros(imagen.rows, imagen.cols, CV_8UC1);
    }

    return imagen;
}


Mat mergeMasks(Mat &imgOriginal, Mat &maskLung, float alpha) {
    Mat imgColor;

    // Convertir a BGR si la original es gris
    if (imgOriginal.channels() == 1)
        cvtColor(imgOriginal, imgColor, COLOR_GRAY2BGR);
    else
        imgColor = imgOriginal.clone();

    // Crear capa verde del mismo tamaño
    Mat greenLayer(imgOriginal.rows, imgOriginal.cols, CV_8UC3, Scalar(0, 255, 0));

    // Crear salida inicial como copia
    Mat output = imgColor.clone();

    // Aplicar overlay SOLO donde la máscara es 255
    for (int i = 0; i < imgOriginal.rows; i++) {
        for (int j = 0; j < imgOriginal.cols; j++) {

            if (maskLung.at<uchar>(i, j) == 255) {

                // output = (1 - alpha) * original + alpha * green
                output.at<Vec3b>(i, j)[0] =
                    (1.0 - alpha) * imgColor.at<Vec3b>(i, j)[0] +
                    alpha * greenLayer.at<Vec3b>(i, j)[0];

                output.at<Vec3b>(i, j)[1] =
                    (1.0 - alpha) * imgColor.at<Vec3b>(i, j)[1] +
                    alpha * greenLayer.at<Vec3b>(i, j)[1];

                output.at<Vec3b>(i, j)[2] =
                    (1.0 - alpha) * imgColor.at<Vec3b>(i, j)[2] +
                    alpha * greenLayer.at<Vec3b>(i, j)[2];
            }
        }
    }

    return output;
}


Mat mergeMasksHeart(Mat &imgOriginal, Mat &maskLung, float alpha ) {
    Mat imgColor;

    // Convertir a BGR si la original es gris
    if (imgOriginal.channels() == 1)
        cvtColor(imgOriginal, imgColor, COLOR_GRAY2BGR);
    else
        imgColor = imgOriginal.clone();

    // Crear capa verde del mismo tamaño
    Mat greenLayer(imgOriginal.rows, imgOriginal.cols, CV_8UC3, Scalar(0, 0, 255));

    // Crear salida inicial como copia
    Mat output = imgColor.clone();

    // Aplicar overlay SOLO donde la máscara es 255
    for (int i = 0; i < imgOriginal.rows; i++) {
        for (int j = 0; j < imgOriginal.cols; j++) {

            if (maskLung.at<uchar>(i, j) == 255) {

                // output = (1 - alpha) * original + alpha * green
                output.at<Vec3b>(i, j)[0] =
                    (1.0 - alpha) * imgColor.at<Vec3b>(i, j)[0] +
                    alpha * greenLayer.at<Vec3b>(i, j)[0];

                output.at<Vec3b>(i, j)[1] =
                    (1.0 - alpha) * imgColor.at<Vec3b>(i, j)[1] +
                    alpha * greenLayer.at<Vec3b>(i, j)[1];

                output.at<Vec3b>(i, j)[2] =
                    (1.0 - alpha) * imgColor.at<Vec3b>(i, j)[2] +
                    alpha * greenLayer.at<Vec3b>(i, j)[2];
            }
        }
    }

    return output;
}

Mat mergeMasksBones(Mat &imgOriginal, Mat &maskBone, float alpha ) {
    Mat imgColor;

    if (imgOriginal.channels() == 1)
        cvtColor(imgOriginal, imgColor, COLOR_GRAY2BGR);
    else
        imgColor = imgOriginal.clone();

    // Color AMARILLO: (Blue=0, Green=255, Red=255)
    Mat yellowLayer(imgOriginal.rows, imgOriginal.cols, CV_8UC3, Scalar(0, 255, 255));

    Mat output = imgColor.clone();

    for (int i = 0; i < imgOriginal.rows; i++) {
        for (int j = 0; j < imgOriginal.cols; j++) {
            if (maskBone.at<uchar>(i, j) == 255) {
                output.at<Vec3b>(i, j)[0] =
                    (1.0 - alpha) * imgColor.at<Vec3b>(i, j)[0] +
                    alpha * yellowLayer.at<Vec3b>(i, j)[0];

                output.at<Vec3b>(i, j)[1] =
                    (1.0 - alpha) * imgColor.at<Vec3b>(i, j)[1] +
                    alpha * yellowLayer.at<Vec3b>(i, j)[1];

                output.at<Vec3b>(i, j)[2] =
                    (1.0 - alpha) * imgColor.at<Vec3b>(i, j)[2] +
                    alpha * yellowLayer.at<Vec3b>(i, j)[2];
            }
        }
    }
    return output;
}


