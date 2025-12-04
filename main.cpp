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
#include <curl/curl.h>

using namespace std;
using namespace cv;
using namespace itk;


// ----- VARIABLES GLOBALES

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

int tiempo = 0;

int kernelSize = 3;

bool clahe = false;
bool eq = false;

bool controles = false;
bool slicer = false;

vector<Mat> imgs;



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

bool corazon = false;
bool hueso= false;
bool pulmones= false;
bool result= false;
bool comparacion= false;
bool extra= false;
bool pruebas= false;


Mat img_normal; // imagen de menu
Mat img_click;  // imagen de menu cuando se cliquea para mostrar la máscara



// Declaración global de botones de la APP (no del menú) -- aplicación de ecualización

Rect btnCLAHE(10, 10, 120, 40);
Rect btnEq(10, 60, 120, 40);

// ... [TUS FUNCIONES AUXILIARES: ITKToMat, readRaw, readIMA, etc. SE MANTIENEN IGUAL] ...
// (Las pego colapsadas para ahorrar espacio, el código lógico no cambia)

void updateKSize(int KSize){ KSize = KSize*2+1; }



// método para obtener una lista de todos los archivos IMA de la carpeta L333 del datasetKaggle
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


// método para leer imagenes .ima y transformarmas a objetos Mat de opencv para manipularlas

Mat readIMA(const string& filename)
{
    // --- Configurar lector DICOM ---
    using ImageIOType = GDCMImageIO;
    ImageIOType::Pointer dicomIO = ImageIOType::New();

    using ReaderType = ImageFileReader<InputImageType>;
    ReaderType::Pointer reader = ReaderType::New();
    reader->SetImageIO(dicomIO);
    reader->SetFileName(filename);
    reader->Update();

    // --- Rescalar intensidades a 0..255 ---
    using RescaleType = RescaleIntensityImageFilter<InputImageType, OutputImageType>;
    RescaleType::Pointer scale = RescaleType::New();
    scale->SetInput(reader->GetOutput());
    scale->SetOutputMinimum(0);
    scale->SetOutputMaximum(255);
    scale->Update();

    OutputImageType::Pointer img = scale->GetOutput();

    // --- Obtener tamaño ---
    auto region   = img->GetLargestPossibleRegion();
    auto size     = region.GetSize();
    int width     = size[0];
    int height    = size[1];

    // --- Crear Mat ---
    Mat out(height, width, CV_8UC1);

    // --- Copiar buffer ITK → OpenCV ---
    unsigned char* buffer = img->GetBufferPointer();
    memcpy(out.data, buffer, width * height);

    return out;
}



// método para activar filtro CLAHE en pruebas
Mat toClahe(Mat imagen){
    Mat salida;
    Ptr<CLAHE> clahe = createCLAHE(2.0, cv::Size(8, 8));
    clahe->apply(imagen, salida);
    return salida;
}

// método para activar Ecualización en pruebas
Mat toEq(Mat imagen){
    equalizeHist(imagen, imagen);
    return imagen;
}

// método para Filtro de Sharpening

Mat applySharpening(Mat input) {
    Mat output = input.clone();

    float g = 1.66f;
    Mat lookUpTable(1, 256, CV_8U);
    uchar* p = lookUpTable.ptr();
    for( int i = 0; i < 256; ++i)
        p[i] = saturate_cast<uchar>(pow(i / 255.0, g) * 255.0);
    LUT(output, lookUpTable, output);

    double sigma = 2.8; 
    double amount = 1.2;

    Mat blurred;
    GaussianBlur(output, blurred, cv::Size(0, 0), sigma);
    addWeighted(output, 1.0 + amount, blurred, -amount, 0, output);

    return output;
}


// método para filtro para rellenar zonas vacías
Mat fillHoles(const Mat& mask) {
    Mat im_floodfill = mask.clone();
    Mat maskFlood;
    copyMakeBorder(im_floodfill, maskFlood, 1, 1, 1, 1, BORDER_CONSTANT, Scalar(0));
    floodFill(maskFlood, cv::Point(0, 0), Scalar(255));
    Mat floodfilled = maskFlood(Rect(1, 1, mask.cols, mask.rows));
    Mat floodfilledInv;
    bitwise_not(floodfilled, floodfilledInv);
    return (mask | floodfilledInv);
}






// método para hacer filtro de umbral, aqui creamos una máscara que considere valores máximos y minimos de intensidad
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

// método para activar Blur gaussiano, y cambiar los valores del kernel y sigma en la ecuacion 
// $exp(-(x+y)/(2*sigma^2))$

Mat toGaussianBlur(Mat imagen, int kernelSize = 5, double sigma = 0) {
    Mat salida;
    if (kernelSize % 2 == 0) kernelSize++;
    GaussianBlur(imagen, salida, cv::Size(kernelSize, kernelSize), sigma);
    return salida;
}




// filtro para seleccionar zona de interés en huesos, eliminar estomago
Mat filterByAreaAndIntensity(const Mat& mask, const Mat& original) {
    Mat labels, stats, centroids;
    int nLabels = connectedComponentsWithStats(mask, labels, stats, centroids, 8, CV_32S);
    Mat out = Mat::zeros(mask.size(), CV_8UC1);

    for (int label = 1; label < nLabels; ++label) {
        int area = stats.at<int>(label, CC_STAT_AREA);

        if (area < 57) continue;

        if (currentSlice >= 245 && currentSlice <= 438) {
            
            double cX = centroids.at<double>(label, 0);
            double cY = centroids.at<double>(label, 1);
            double termX = std::pow(cX - 260, 2) / std::pow(127.0, 2);
            double termY = std::pow(cY - 217, 2) / std::pow(94.0, 2);

            if ((termX + termY) <= 1.0) {
                if (area < 13187) {
                    continue; 
                }
            }
        }

        Mat componentMask = (labels == label);
        Scalar meanVal = mean(original, componentMask);
        if (meanVal[0] < 83) continue;

        out.setTo(255, componentMask);
    }
    return out;
}


// método para Generar un circulo de máscara, nos sirve para tomar los valores que deseamos e ignorar lo demás, basicamente 
// marcar la región de interés
Mat maskCircle(Mat &img, int cx, int cy, int rx, int ry) {
    Mat mask = Mat::zeros(img.rows, img.cols, CV_8UC1);
    ellipse(mask, cv::Point(cx, cy), cv::Size(rx, ry), 0, 0, 360, Scalar(255), FILLED);
    Mat salida(img.rows, img.cols, img.type(), Scalar(255));
    
    img.copyTo(salida, mask);
    return salida;
}
Mat maskCircle2(Mat &img, int cx, int cy, int rx, int ry) {
    Mat mask = Mat::zeros(img.rows, img.cols, CV_8UC1);

    // 1. Dibujar círculo blanco = zona visible
    ellipse(mask, cv::Point(cx, cy), cv::Size(rx, ry), 0, 0, 360, Scalar(255), FILLED);

    // 2. Crear salida blanca (color blanco completo)   
    Mat salida = Mat::zeros(img.size(), img.type());

    // 3. Copiar dentro del círculo la imagen original
    img.copyTo(salida, mask);

    return salida;
}
// método para hacer erosión y dilatación
Mat open(Mat imagen, int kernelSize){
    Mat erodida, salida;
    Mat kernel = getStructuringElement(MORPH_RECT, cv::Size(kernelSize, kernelSize));
    erode(imagen, erodida, kernel);
    dilate(erodida, salida, kernel);
    return salida;
}

// método para hacer dilatación y erosión
Mat close(Mat imagen, int kernelSize){
    Mat dilatada, salida;
    Mat kernel = getStructuringElement(MORPH_RECT, cv::Size(kernelSize, kernelSize));
    dilate(imagen, dilatada, kernel);
    erode(dilatada, salida, kernel);
    return salida;
}


//  definición de métodos, para poder usarlos previo a su creación

Mat mejorarNitidez(Mat imagenEntrada);



//MÉTODO PARA LA "interfaz gráfica" para saber si el click se dio adentro o fuera de un botón

bool inside(Button b, int x, int y) {
    return (x > b.x1 && x < b.x2 && y > b.y1 && y < b.y2);
}


// método para dibujar la interfaz cuando se presiona un botón, permite marcar en donde se presionó generando una máscara

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



void encenderVentana(int boton){
    tiempo = 1;
    corazon = false;
    hueso= false;
    pulmones= false;
    result= false;
    comparacion= false;
    extra= false;
    pruebas= false;

    if(boton  == 1){
        corazon = true;
    } else if(boton  == 2){
        hueso = true;
    } else if(boton  == 3){
        pulmones = true;
    } else if(boton  == 4){
        result = true;
    } else if(boton  == 5){
        comparacion = true;
    } else if(boton  == 6){
        extra = true;
    } else if(boton  == 7){
        pruebas = true;
    }
}

// --- EVENTOS DE CLICK --- en el menú principal

void onMouse(int event, int x, int y, int flags, void* userdata) {
    if (event == EVENT_LBUTTONDOWN) {

        // Copiamos la imagen normal
        img_click = img_normal.clone();
        // ---- BOTON CORAZÓN ----
        if (inside(btn_corazon, x, y)) {
            drawMask(img_click, btn_corazon);
            imshow("Aplicacion Principal", img_click);
            waitKey(120);   // pequeña pausa visual
            encenderVentana(1);
        }

        // ---- BOTON HUESOS ----
        else if (inside(btn_huesos, x, y)) {
            drawMask(img_click, btn_huesos);
            imshow("Aplicacion Principal", img_click);
            waitKey(120);
            cout << "Botón: HUESOS\n";
            encenderVentana(2);
        }

        // ---- BOTON PULMONES ----
        else if (inside(btn_pulmones, x, y)) {
            drawMask(img_click, btn_pulmones);
            imshow("Aplicacion Principal", img_click);
            waitKey(120);
            cout << "Botón: PULMONES\n";
            encenderVentana(3);
        }

        // ---- RESULTADOS ----
        else if (inside(btn_result, x, y)) {
            drawMask(img_click, btn_result);
            imshow("Aplicacion Principal", img_click);
            waitKey(120);
            cout << "Botón: RESULTADOS\n";
            encenderVentana(4);
        }

        // ---- COMPARACION DnCNN ----
        else if (inside(btn_comp, x, y)) {
            drawMask(img_click, btn_comp);
            imshow("Aplicacion Principal", img_click);
            waitKey(120);
            cout << "Botón: COMPARACION DnCNN\n";
            encenderVentana(5);
        }

        // ---- EXTRA ----
        else if (inside(btn_extra, x, y)) {
            drawMask(img_click, btn_extra);
            imshow("Aplicacion Principal", img_click);
            waitKey(120);
            cout << "Botón: EXTRA\n";
            encenderVentana(6);
        }

        // ---- PRUEBAS ----
        
        else if (inside(btn_pruebas, x, y)) {
            drawMask(img_click, btn_pruebas);
            imshow("Aplicacion Principal", img_click);
            waitKey(120);
            cout << "Botón: PRUEBAS\n";
            encenderVentana(7);
        }

        // Restauramos la imagen normal después de la animación
        imshow("Aplicacion Principal", img_normal);
    }
}




Mat boneWindowing(Mat imagen, int minVal, int maxVal) {
    Mat salida = imagen.clone();
    double escala = 255.0 / (double)(maxVal - minVal);
    imagen.convertTo(salida, -1, escala, -minVal * escala);
    return salida;
}


Mat defineBones(Mat imagen) {

    imagen = maskCircle2(imagen, 253, 264, 210, 147);

    Mat work = imagen.clone();
    if (work.type() != CV_8UC1) work.convertTo(work, CV_8UC1);

    // 1. MEJORA DE CONTRASTE (Siempre Sharpen)
    work = applySharpening(work);

    // 2. Blur (Kernel 3x3 fijo)
    GaussianBlur(work, work, cv::Size(3, 3), 0);

    // 3. Umbral (117 a 255 fijos)
    Mat mask = Umbrilize(work, 117, 255);

    // 4. Morfología básica 
    mask = close(mask, 3);
    
    // 5. Cerrar Gaps (Puenteo Agresivo)

    int kGap = 21; 
    Mat k = getStructuringElement(MORPH_ELLIPSE, cv::Size(kGap, kGap));
    dilate(mask, mask, k);
    mask = fillHoles(mask);
    erode(mask, mask, k);

    // 6. Filtro Dureza (Area/Media/ZonaMuerte)
    Mat maskFiltrada = filterByAreaAndIntensity(mask, work);

    // 7. Rellenar huecos finales
    maskFiltrada = fillHoles(maskFiltrada);

    return maskFiltrada;
}



Mat mergeAllMasks(
    Mat &imgOriginal,
    Mat *maskLung,   // verde
    Mat *maskHeart,  // rojo
    Mat *maskBone,   // amarillo
    float alpha = 0.3 //valor de transparencia
) 
{
    // Convertir a color si es necesario
    Mat imgColor;
    if (imgOriginal.channels() == 1)
        cvtColor(imgOriginal, imgColor, COLOR_GRAY2BGR);
    else
        imgColor = imgOriginal.clone();

    Mat output = imgColor.clone();

    // Capas de color para cada máscara
    Scalar colorLung  = Scalar(0, 255, 0);   // verde
    Scalar colorHeart = Scalar(0, 0, 255);   // rojo
    Scalar colorBone  = Scalar(0, 255, 255); // amarillo

    int rows = imgColor.rows;
    int cols = imgColor.cols;

    // Recorrer píxeles una sola vez
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {

            Vec3b orig = imgColor.at<Vec3b>(i, j);
            Vec3b &dst = output.at<Vec3b>(i, j);

            // Copia inicial
            float r = orig[2], g = orig[1], b = orig[0];

            // LUNG
            if (maskLung && !maskLung->empty() && maskLung->at<uchar>(i, j) == 255) {
                b = (1 - alpha) * b + alpha * colorLung[0];
                g = (1 - alpha) * g + alpha * colorLung[1];
                r = (1 - alpha) * r + alpha * colorLung[2];
            }

            // HEART
            if (maskHeart && !maskHeart->empty() && maskHeart->at<uchar>(i, j) == 255) {
                b = (1 - alpha) * b + alpha * colorHeart[0];
                g = (1 - alpha) * g + alpha * colorHeart[1];
                r = (1 - alpha) * r + alpha * colorHeart[2];
            }

            // BONE
            if (maskBone && !maskBone->empty() && maskBone->at<uchar>(i, j) == 255) {
                b = (1 - alpha) * b + alpha * colorBone[0];
                g = (1 - alpha) * g + alpha * colorBone[1];
                r = (1 - alpha) * r + alpha * colorBone[2];
            }

            dst = Vec3b((uchar)b, (uchar)g, (uchar)r);
        }
    }

    return output;
}
// MÉTODO CON LOS VALORES QUE OBTUVIMOS MEJORES PARA DEFINIR EL CORAZÓN, Y LOS PULMONES

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

void visualizeStats(Mat& imgDisplay, const Mat& mask, const Mat& imgOriginal) {
    Mat labels, stats, centroids;
    int nLabels = connectedComponentsWithStats(mask, labels, stats, centroids, 8, CV_32S);

    for(int i = 1; i < nLabels; i++) {
        // Area (Morfología)
        int area = stats.at<int>(i, CC_STAT_AREA);
        if(area < 50) continue; // Ignorar ruido muy pequeño para no saturar texto

        // Mascara individual para calcular intensidad
        Mat objMask = (labels == i);
        Mat objMask8u;
        objMask.convertTo(objMask8u, CV_8UC1);
        
        // Estadisticas de Intensidad (Densidad)
        Scalar meanVal, stdDevVal;
        meanStdDev(imgOriginal, meanVal, stdDevVal, objMask8u);

        // Centroide para poner el texto
        double cX = centroids.at<double>(i, 0);
        double cY = centroids.at<double>(i, 1);

        // Formato del texto: "A:Area M:MediaIntensidad"
        // A = Tamaño hueso, M = Densidad aproximada
        string info = "A:" + to_string(area) + " M:" + to_string((int)meanVal[0]);
        
        // Dibujar texto rojo pequeño. CORREGIDO: cv::Point
        putText(imgDisplay, info, cv::Point(cX - 20, cY), FONT_HERSHEY_SIMPLEX, 0.35, Scalar(0, 0, 255), 1);
    }
}




Mat sumarMascaras(Mat *maskLung, Mat *maskHeart, Mat *maskBone)
{
    // Crear máscara final vacía
    Mat finalMask = *maskLung;

    bitwise_or(finalMask, *maskHeart, finalMask);
    bitwise_or(finalMask, *maskBone, finalMask);

    return finalMask;
}

double getMemoryUsageMB() {
    ifstream file("/proc/self/status");
    string line;
    double result = 0;
    while (getline(file, line)) {
        if (line.substr(0, 6) == "VmRSS:") {
            stringstream ss(line);
            string temp;
            long usageKB;
            ss >> temp >> usageKB; // Saltamos "VmRSS:" y leemos el numero
            result = usageKB / 1024.0;
            break;
        }
    }
    return result;
}




///INTERFACE METHODS, NOT IMPLEMENTED TILL THE END
void encenderCorazon(Mat original);
void encenderHueso(Mat original);
void encenderPulmones(Mat original);
void encenderResults(Mat original);
void encenderComparativa(Mat original);
void encenderExtra(Mat original);
void encenderPruebas(Mat &imagenBase);

// MAIN

int main() {
    // menu principal
    img_normal = imread("img_normal.png");
    resize(img_normal,img_normal,cv::Size(500, 395));
    namedWindow("Aplicacion Principal", WINDOW_AUTOSIZE);
    setMouseCallback("Aplicacion Principal", onMouse);
    moveWindow("Aplicacion Principal", 0,0); // la posicionamos en 0,0 para ajustarlo despues
    
    
    string folder = "L333"; // carpeta de imagenes
    vector<string> files = getIMA(folder);   // lista de imagenes (.ima ) string

    imgs;

    imgs.reserve(files.size());// necesario para optimizar el uso de memoria


    for (auto& f : files) { // almacenamiento imagen a imagen de los archivos de la carpeta
        Mat m = readIMA(f);
        if (m.empty()) m = Mat::zeros(256, 256, CV_8UC1);
        imgs.push_back(m);
    }  

        cout << "\n[INFO] Carga completada. RAM usada estimada: " << getMemoryUsageMB() << " MB" << endl;

    // createTrackbar("Slice", "Aplicacion Principal", &currentSlice, imgs.size() - 1);

    // VENTANAS
    while (true) {
        
        // MOSTRAMOS EL MENÚ, SIEMPRE ES VISIBLE
        
               Mat menuDisplay = img_normal.clone();

        // 2. Obtener RAM actual
        double currentRam = getMemoryUsageMB();
        string ramText = "RAM: " + to_string((int)currentRam) + " MB";

        // 3. Dibujar texto en la esquina inferior izquierda (amarillo con sombra negra)
        putText(menuDisplay, ramText, cv::Point(10, 380), FONT_HERSHEY_SIMPLEX, 0.6, Scalar(0, 0, 0), 2);
        putText(menuDisplay, ramText, cv::Point(10, 380), FONT_HERSHEY_SIMPLEX, 0.6, Scalar(0, 255, 255), 1);

        // 4. Mostrar el menú modificado
        imshow("Aplicacion Principal", menuDisplay);
        
        if (imgs.empty()) {
            if (waitKey(30) == 27) break;
            continue; 
        }

        // Validación de slice seguro
        if (currentSlice < 0) currentSlice = 0;
        if (currentSlice >= imgs.size()) currentSlice = imgs.size() - 1;

        Mat original = imgs[currentSlice].clone();
        if(corazon){// PROCESO COMPLETO PARA OBTENER MASCARA DE CORAZON
            encenderCorazon(original);

        } else if (hueso){ // PROCESO COMPLETO PARA OBTENER MASCARA DE HUESOS
            encenderHueso(original);

        } else if (pulmones){ // PROCESO COMPLETO PARA OBTENER MASCARA DE PULMONES
            encenderPulmones(original);
            
        } else if (result){ // RESULTADOS DE LA APLICACIÓN
            encenderResults(original);
            
        } else if (comparacion){ // COMPARACIÓN DE RESULTADOS Y BLUR GAUSSIANO CON LA RED DnCNN EN PYTHON
            encenderComparativa(original);
            
        } else if (extra){ // VALORES DE LAS IMAGENES AISLADAS Y RESALTADAS EN LAS ZONAS QUE NOS INTERESAN
            // encenderExtra(original);
            
        } else if (pruebas){ // SLIDERS Y BOTONES PARA HACER PRUEBAS CON LAS IMAGENES
            encenderPruebas(imgs[currentSlice]);
            pruebas = false;   // Cerrar y volver al menú
            
        }
        
        if (waitKey(30) == 27) break; // ESC para salir
    }

    destroyAllWindows();
    return 0;
}





// para almacenar respuesta de cURL
size_t WriteCallback(void* contents, size_t size, size_t nmemb, std::vector<uchar>* userp)
{
    size_t total = size * nmemb;
    uchar* data = (uchar*)contents;
    userp->insert(userp->end(), data, data + total);
    return total;
}


Mat sendForDenoise(Mat img)
{
    // --- codificar la imagen como PNG ---
    std::vector<uchar> buf;
    imencode(".png", img, buf);

    // --- Inicializar CURL ---
    CURL* curl = curl_easy_init();
    if (!curl) return img.clone();

    std::vector<uchar> response;

    curl_mime* mime;
    curl_mimepart* part;

    mime = curl_mime_init(curl);

    // archivo enviado
    part = curl_mime_addpart(mime);
    curl_mime_name(part, "file");
    curl_mime_filename(part, "input.png");
    curl_mime_data(part, (const char*)buf.data(), buf.size());

    // configurar CURL
    curl_easy_setopt(curl, CURLOPT_URL, "http://0.0.0.0:8000/denoise");
    curl_easy_setopt(curl, CURLOPT_MIMEPOST, mime);
    curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, WriteCallback);
    curl_easy_setopt(curl, CURLOPT_WRITEDATA, &response);

    // ejecutar
    CURLcode res = curl_easy_perform(curl);

    Mat out;

    if (res == CURLE_OK) {
        out = imdecode(response, IMREAD_GRAYSCALE);
    } else {
        std::cerr << "ERROR denoise: " << curl_easy_strerror(res) << std::endl;
        out = img.clone();
    }

    curl_mime_free(mime);
    curl_easy_cleanup(curl);

    return out;
}

void exportData(string organName) {
    string filename = organName + "_estadisticas.csv";
    cout << "\n[INFO] Iniciando exportacion de datos a: " << filename << "..." << endl;
    
    ofstream file(filename);
    if(!file.is_open()) {
        cout << "[ERROR] No se pudo crear el archivo csv." << endl;
        return;
    }
       file << "Slice,Area_Total(px),Densidad_Media(HU_prox)\n";
       int backupSlice = currentSlice;

  for (size_t i = 0; i < imgs.size(); i++) {
        // Actualizamos variable global para que los filtros funcionen bien (ej. zona muerte)
        currentSlice = (int)i; 
        
        Mat original = imgs[i].clone();
        Mat mask;

        // Ejecutar la segmentación correspondiente
        if (organName == "Huesos") {
            mask = defineBones(original);
        } else if (organName == "Corazon") {
            mask = defineOrgan(original, false);
        } else if (organName == "Pulmones") {
            mask = defineOrgan(original, true);
        }
        int areaTotal = countNonZero(mask);
        double meanVal = 0.0;
        if (areaTotal > 0) {
            Scalar m, s;
            meanStdDev(original, m, s, mask);
            meanVal = m[0];
        }
        file << i << "," << areaTotal << "," << meanVal << "\n";
   
        if (i % 10 == 0) cout << "Procesando slice " << i << "/" << imgs.size() << "\r" << flush;
    }
    currentSlice = backupSlice;
    file.close();
    cout << "\n[EXITO] Exportacion completada. Revisa la carpeta del proyecto." << endl;
}





// INTERFACES

void eliminarControles(){ 
    destroyWindow("Aplicacion Principal"); 
    namedWindow("Aplicacion Principal", WINDOW_AUTOSIZE); 
    setMouseCallback("Aplicacion Principal", onMouse); 
    createTrackbar("Slice", "Aplicacion Principal", &currentSlice, imgs.size() - 1); 
    moveWindow("Aplicacion Principal", 0,0); // la posicionamos en 0,0 para ajustarlo despues controles = false; 
}

void createSliceTrackbar() {
    int maxVal = imgs.empty() ? 0 : (int)imgs.size() - 1;
    if (currentSlice > maxVal) currentSlice = maxVal;
    if (currentSlice < 0) currentSlice = 0;

    createTrackbar("Slice", "Aplicacion Principal", &currentSlice, maxVal);
}
// Cerrar ventanas auxiliares
void cerrarVentanas() {
    static vector<string> windows = {
        "Original", "Region de interés", "Umbralización", "Apertura",
        "Sharpening", "Cierre", "CLAHE", "EQ", "Gauss", "Resultado",
        "Cierre2", "Mascara Filtrada", "Aislada", "PRUEBAS", "ORIGINAL",
        "BLUR", "UMBRAL", "CLAHE", "EQUALIZE", "CIRCULAR MASK","FINAL","Denoised (Python)"
    };
    // destroyWindow("Aislada");
    
    for (auto &w : windows) destroyWindow(w);
    
}


// Resetear o activar controles (caso Pruebas)
void enableControls() {
    controles = true;
    // Aquí pones los trackbars de pruebas
}


// Para interfaces normales (corazón, pulmones, hueso, etc.)
void prepareStandardView() {
    if (controles) eliminarControles();

    if (tiempo == 1) {
        cerrarVentanas();

        if (!slicer) {
            createSliceTrackbar();
            slicer = true;
        }

        tiempo = 2;
    }
}

void encenderCorazon(Mat original){

    prepareStandardView();

    int min = 110;
    int max = 170;
    int kernel = 3;
    double sigma = 1;

    int cxdf = 284;
    int cydf = 220;
    int radioXdf = 87;
    int radioYdf = 65;
    
    Mat roi;
    if(currentSlice <= 41){
        roi =  maskCircle(original, cxdf, cydf, radioXdf, radioYdf);
    } else {
        roi = Mat::ones(original.rows, original.cols, CV_8UC1);
    }

    Mat blur = toGaussianBlur(roi, kernel, sigma);
    
    Mat umbral = Umbrilize(blur, min, max);

    Mat apertura = open(umbral, kernel);

    Mat cierre = close(apertura, kernel);


    vector<vector<cv::Point>> contours;
    vector<Vec4i> hierarchy;
    // Encontrar contornos externos
    findContours(cierre, contours, hierarchy, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

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
    
    Mat cleanMask = Mat::zeros(cierre.size(), CV_8UC1);
    
    if (maxAreaIdx >= 0) {
        drawContours(cleanMask, contours, maxAreaIdx, Scalar(255), FILLED);
    }
    
    Mat cierre2 = cleanMask;
    Mat statsDisplay = roi.clone();
    cvtColor(statsDisplay, statsDisplay, COLOR_GRAY2BGR);
    visualizeStats(statsDisplay, cierre2, original); 

     putText(statsDisplay, "[G] Generar Reporte CSV", cv::Point(10, 30), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0,255,0), 2);
   
    imshow("Original", original);
    moveWindow("Original", 500,0);
    imshow("Region de interés", statsDisplay);
    moveWindow("Region de interés", 500+(original.cols)*1,0);
    imshow("Umbralización", umbral);
    moveWindow("Umbralización", 500+(original.cols)*2,0);
    imshow("Apertura", apertura);
    moveWindow("Apertura", 500+(original.cols)*0,original.rows);
    imshow("Cierre", cierre);
    moveWindow("Cierre", 500+(original.cols)*1,original.rows);
    imshow("Cierre2", cierre2);
    moveWindow("Cierre2", 500+(original.cols)*2,original.rows);

    int ke = waitKey(120);
    if (ke == 'g' || ke == 'G') exportData("Pulmones");
}

void encenderHueso(Mat original){
    prepareStandardView();

    Mat work = original.clone();

    Mat roi = maskCircle2(work, 253, 264, 210, 147);

    Mat sharp = applySharpening(roi);
    Mat blurred;
    GaussianBlur(sharp, blurred, cv::Size(3,3), 0);
    Mat umbral = Umbrilize(blurred, 117, 255);
    Mat cierre = close(umbral, 3);

    int kGap = 21;
    Mat k = getStructuringElement(MORPH_ELLIPSE, cv::Size(kGap, kGap));
    Mat flood;
    dilate(cierre, flood, k);
    flood = fillHoles(flood);
    erode(flood, flood, k);

    Mat maskFiltrada = filterByAreaAndIntensity(flood, roi);

    Mat Cierre2 = fillHoles(maskFiltrada);

    Mat statsDisplay = roi.clone();
    cvtColor(statsDisplay, statsDisplay, COLOR_GRAY2BGR);
    visualizeStats(statsDisplay, Cierre2, original); 

     putText(statsDisplay, "[G] Generar Reporte CSV", cv::Point(10, 30), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0,255,0), 2);
    
    imshow("Original", original);
    moveWindow("Original", 500+(original.cols)*0,0);
    imshow("Region de interés", statsDisplay);
    moveWindow("Region de interés", 500+(original.cols)*1,0);
    imshow("Sharpening", sharp);
    moveWindow("Sharpening", 500+(original.cols)*2,0);
    imshow("Umbralización", umbral);
    moveWindow("Umbralización", 500+(original.cols)*0,original.rows);
    imshow("Cierre", cierre);
    moveWindow("Cierre", 500+(original.cols)*1,original.rows);
    imshow("Mascara Filtrada", maskFiltrada);
    moveWindow("Mascara Filtrada", 500+(original.cols)*2,original.rows);
    imshow("Cierre2", Cierre2);
    moveWindow("Cierre2", 0,original.rows);

    int ke = waitKey(120);
    if (ke == 'g' || ke == 'G') exportData("Huesos");
    

}

void encenderPulmones( Mat original){
    prepareStandardView();

    int cxdf, cydf, radioXdf, radioYdf;
    int mindf, maxdf, kerneldf;
    int sliceLimit;

    double sigmadf = 1;
    
    cxdf = 253;
    cydf = 264;
    radioXdf = 210;
    radioYdf = 147;
    mindf = 0;
    maxdf = 58;
    kerneldf = 9;
    sliceLimit = 127; 
    Mat roi;
    Mat blur;
    Mat umbral;
    Mat apertura;

    if(currentSlice <= 127){
        roi = maskCircle(original, cxdf, cydf, radioXdf, radioYdf);
        blur = toGaussianBlur(roi,kerneldf,sigmadf);
        umbral = Umbrilize(blur, mindf, maxdf);
        apertura = open(umbral, kernelSize);
    } else {
        roi = Mat::zeros(original.rows, original.cols,CV_8UC1);
        blur = roi;
        umbral = roi;
        umbral = roi;
        apertura = roi;
    }

     Mat statsDisplay = roi.clone();
    cvtColor(statsDisplay, statsDisplay, COLOR_GRAY2BGR);
    visualizeStats(statsDisplay, apertura, original);

     putText(statsDisplay, "[G] Generar Reporte CSV", cv::Point(10, 30), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0,255,0), 2);
   
    imshow("Original", original);
    moveWindow("Original", 500+(original.cols)*0,0);
    imshow("Region de interés", statsDisplay);
    moveWindow("Region de interés", 500+(original.cols)*1,0);
    imshow("Blur Gaussiano", blur);
    moveWindow("Blur Gaussiano", 500+(original.cols)*2,0);
    imshow("Umbralización", umbral);
    moveWindow("Umbralización", 500+(original.cols)*0,original.rows);
    imshow("Apertura", apertura);
    moveWindow("Apertura", 500+(original.cols)*1,original.rows);

    // --- DETECTAR TECLA G ---
    int ke = waitKey(120);
    if (ke == 'g' || ke == 'G') exportData("Pulmones");

}
void encenderResults(Mat original){
    prepareStandardView();

    imshow("Original", original);
    moveWindow("Original", 500+(original.cols)*0,0);
    Mat original1 = original.clone();
    Mat original2 = original.clone();
    Mat original3 = original.clone();
    Mat corazon = defineOrgan(original1,false);
    Mat pulmones = defineOrgan(original2,true);
    Mat huesos = defineBones(original3);
    
    Mat pulmones2 = pulmones.clone();
    Mat merged = mergeAllMasks(original, &pulmones2, &corazon, &huesos, 0.3);
    Mat binaria = sumarMascaras(&pulmones2, &corazon, &huesos);
    Mat isolated;
    bitwise_and(original, binaria, isolated);

    imshow("Original", original);
    moveWindow("Original", 500+(original.cols)*0,0);
    imshow("Corazon", corazon);
    moveWindow("Corazon", 500+(original.cols)*1,0);
    imshow("Pulmones", pulmones);
    moveWindow("Pulmones", 500+(original.cols)*2,0);
    imshow("Huesos", huesos);
    moveWindow("Huesos", 500+(original.cols)*0,original.rows);
    imshow("Mezcla", merged);
    moveWindow("Mezcla", 500+(original.cols)*1,original.rows);
    imshow("Aislada", isolated);
    moveWindow("Aislada", 500+(original.cols)*2,original.rows);
    
    
    
}
void encenderComparativa(Mat original){

    prepareStandardView();

    Mat original1 = original.clone();

    Mat gauss = toGaussianBlur(original1, 3);

    Mat denoised = sendForDenoise(original);

    imshow("Original", original);
    moveWindow("Original", 500,0);

    imshow("Denoise Gauss", gauss);
    moveWindow("Denoise Gauss", 0,500);

    imshow("Denoised (Python)", denoised);
    moveWindow("Denoised (Python)", original.cols*2 + 40, 0);

    waitKey(10); // muestra ventana temporal
}




// void encenderExtra(Mat original){
//     prepareStandardView();

//     ///  TODO: Quitar controles
//     ///  TODO: imagen con los organos aislados
//     ///  TODO: imagenes con los pulmones resaltados, dado la opinion del radiologo
//     ///  TODO: imagen merge, de los organos realzados
    
// }

void encenderPruebas(Mat &imagenBase){
    controles = true;

    namedWindow("PRUEBAS", WINDOW_NORMAL);
    resizeWindow("PRUEBAS", 600, 400);

    // --- TRACKBARS ---
    createTrackbar("Kernel Blur",      "PRUEBAS", &ksizeTrack, 20);
    createTrackbar("Sigma Blur x10",   "PRUEBAS", &sigmaTrack, 100);
    createTrackbar("Umbral Min",       "PRUEBAS", &umbralMin, 255);
    createTrackbar("Umbral Max",       "PRUEBAS", &umbralMax, 255);
    createTrackbar("CLAHE (0/1)",      "PRUEBAS", (int*)&clahe, 1);
    createTrackbar("Equalize (0/1)",   "PRUEBAS", (int*)&eq, 1);
    createTrackbar("cx", "PRUEBAS", &cx, imagenBase.cols);
    createTrackbar("cy", "PRUEBAS", &cy, imagenBase.rows);
    createTrackbar("radioX", "PRUEBAS", &radioX, imagenBase.cols/2);
    createTrackbar("radioY", "PRUEBAS", &radioY, imagenBase.rows/2);

    // --- Ventanas de depuración ---
    namedWindow("ORIGINAL", WINDOW_NORMAL);
    namedWindow("BLUR", WINDOW_NORMAL);
    namedWindow("UMBRAL", WINDOW_NORMAL);
    namedWindow("CLAHE", WINDOW_NORMAL);
    namedWindow("EQUALIZE", WINDOW_NORMAL);
    namedWindow("CIRCULAR MASK", WINDOW_NORMAL);
    namedWindow("FINAL", WINDOW_NORMAL);


    Mat step_blur, step_umbral, step_clahe, step_eq, step_circular;

    while (true) {

        Mat preview = imagenBase.clone();

        // -----------------------------------
        // 1. BLUR
        // -----------------------------------
        int kB = (ksizeTrack % 2 == 0 ? ksizeTrack+1 : ksizeTrack);
        double sigma = sigmaTrack / 10.0;

        if (kB >= 1)
            step_blur = toGaussianBlur(preview, kB, sigma);
        else
            step_blur = preview.clone();

        // -----------------------------------
        // 2. UMBRAL
        // -----------------------------------
        step_umbral = Umbrilize(step_blur, umbralMin, umbralMax);

        // -----------------------------------
        // 3. CLAHE
        // -----------------------------------
        step_clahe = clahe ? toClahe(step_blur) : step_blur.clone();

        // -----------------------------------
        // 4. EQUALIZE
        // -----------------------------------
        step_eq = eq ? toEq(step_clahe) : step_clahe.clone();

        // -----------------------------------
        // 5. Máscara circular
        // -----------------------------------
        step_circular = maskCircle2(step_eq, cx, cy, radioX, radioY);

        // -----------------------------------
        // MOSTRAR TODAS LAS VENTANAS
        // -----------------------------------
        imshow("ORIGINAL", imagenBase);
        moveWindow("ORIGINAL", 0, 0);

        imshow("BLUR", step_blur);
        moveWindow("BLUR", imagenBase.cols + 10, 0);

        imshow("UMBRAL", step_umbral);
        moveWindow("UMBRAL", (imagenBase.cols * 2) + 20, 0);

        imshow("CLAHE", step_clahe);
        moveWindow("CLAHE", 0, imagenBase.rows + 40);

        imshow("EQUALIZE", step_eq);
        moveWindow("EQUALIZE", imagenBase.cols + 10, imagenBase.rows + 40);

        imshow("CIRCULAR MASK", step_circular);
        moveWindow("CIRCULAR MASK", (imagenBase.cols * 2) + 20, imagenBase.rows + 40);

        imshow("FINAL", step_circular);
        moveWindow("FINAL", imagenBase.cols, imagenBase.rows * 2 + 80);

        // -----------------------------------
        // Escape
        int key = waitKey(30);
        if (key == 27) break;
    }

    // Cerrar ventanas
    destroyWindow("PRUEBAS");
    destroyWindow("ORIGINAL");
    destroyWindow("BLUR");
    destroyWindow("UMBRAL");
    destroyWindow("CLAHE");
    destroyWindow("EQUALIZE");
    destroyWindow("CIRCULAR MASK");
    destroyWindow("FINAL");
}
