#include <algorithm>
#include <memory>
#include "iostream"

#include "../tensorNet.h"
#include "../util/cuda/cudaUtility.h"
#include "../util/cuda/cudaMappedMemory.h"

#include <opencv2/opencv.hpp>


//static const int TIMING_ITERATIONS = 100;

class NetworkModel{
public:
    NetworkModel(){

        // загрузка имен классов объектов
        this->labels = loadLabelInfo(label_file);

        // создание модели
        this->tensorNet.caffeToTRTModel(this->model_file, this->weight_file, this->layers_name, this->batch_size);
        this->tensorNet.createInference();

        // выделяем память для слоев
        this->allocate_layers();

        this->buffers = new float*[this->buf_layers.size()];
        this->set_buffers();
    }

    ~NetworkModel(){
        delete []this->buffers;
    }

    // получение памяти слоя data
    float*  get_buf_layer_data(){
        return this->buf_layers.at(0);
    }

    // получаем измерение входного слоя data объявленное в прототипе сети
    nvinfer1::DimsCHW get_dims_data(){
        return this->tensorNet.getTensorDims(this->layers_name.at(0));
    }

    // заполнение буфера для передачи в IExecutionContext.execute
    void set_buffers(){
        for(int i=0; i<this->buf_layers.size(); i++){
            this->buffers[i]=this->buf_layers[i];
        }
    }

    // детектирование изображения
    void image_inference(){
        this->tensorNet.imageInference((void**)this->buffers, this->buf_layers.size(), this->batch_size);
    }

    // структура для хранения аттрибутов слоя-вывода
    struct detection_out_struct {
        float image_id, label, score, xmin, ymin, xmax, ymax;
    };

    // наполнить вектор аттрибутами распознанного изображения
    void set_detection_out(){
        detection_out_struct * out = (detection_out_struct*) this->buf_layers.back();
        for(int i=0; i < this->tensorNet.getTensorDims(this->layers_name.back()).h() ; i++){
            if (out[i].label != -1){
                this->detection_out.push_back(&out[i]);
            }
        }
    }

    std::vector<detection_out_struct*> detection_out;   // вектор указателей на указатели структур аттрибутов распознанного изображения
    std::vector<std::string> labels;                    // вектор содержит имена классов объектов


private:
    // выделение памяти для слоев
    float* allocateMemory(DimsCHW dims)
    {
        float* ptr;
        size_t size;
        size = this->batch_size * dims.c() * dims.h() * dims.w();
        assert(!cudaMallocManaged(&ptr, size*sizeof(float)));
        return ptr;
    }

    // load label info
    std::vector<std::string> loadLabelInfo(const char* filename)
    {
        assert(filename);
        std::vector<std::string> labelInfo;

        FILE* f = fopen(filename, "r");
        if( !f )
        {
            printf("failed to open %s\n", filename);
            assert(0);
        }

        char str[512];
        while( fgets(str, 512, f) != NULL )
        {
            const int syn = 9;  // length of synset prefix (in characters)
            const int len = strlen(str);

            if( len > syn && str[0] == 'n' && str[syn] == ' ' )
            {
                str[syn]   = 0;
                str[len-1] = 0;

                const std::string b = (str + syn + 1);
                labelInfo.push_back(b);
            }
            else if( len > 0 )      // no 9-character synset prefix (i.e. from DIGITS snapshot)
            {
                if( str[len-1] == '\n' ) str[len-1] = 0;
                labelInfo.push_back(str);
            }
        }
        fclose(f);
        return labelInfo;
    }

    // выделение памяти
    void allocate_layers(){
        for(auto el:this->layers_name){
            nvinfer1::DimsCHW dims = this->tensorNet.getTensorDims(el);       // принимаем значение измерения слоя
            this->buf_layers.push_back(this->allocateMemory(dims)); // выделяем память для слоя
        }
    }

    int batch_size = 1;
    const char * model_file = "/home/ubuntu/CODE/SSD/original_models/VGGNet/VOC0712Plus/SSD_300x300_ft/deploy.prototxt";
    const char * weight_file = "/home/ubuntu/CODE/SSD/original_models/VGGNet/VOC0712Plus/SSD_300x300_ft/VGG_VOC0712Plus_SSD_300x300_iter_240000.caffemodel";
    const char * label_file  = "/home/ubuntu/CODE/SSD/original_models/VGGNet/VOC0712Plus/SSD_300x300_ft/labels.txt";

    // имена слоев из модели не поддерживаемых TensorRT
    std::vector<std::string> layers_name={
        "data",
        "conv4_3_norm",
        "conv4_3_norm_mbox_loc_perm",
        "conv4_3_norm_mbox_loc_flat",
        "conv4_3_norm_mbox_conf_perm",
        "conv4_3_norm_mbox_conf_flat",
        "conv4_3_norm_mbox_priorbox",
        "fc7_mbox_loc_perm",
        "fc7_mbox_loc_flat",
        "fc7_mbox_conf_perm",
        "fc7_mbox_conf_flat",
        "fc7_mbox_priorbox",
        "conv6_2_mbox_loc_perm",
        "conv6_2_mbox_loc_flat",
        "conv6_2_mbox_conf_perm",
        "conv6_2_mbox_conf_flat",
        "conv6_2_mbox_priorbox",
        "conv7_2_mbox_loc_perm",
        "conv7_2_mbox_loc_flat",
        "conv7_2_mbox_conf_perm",
        "conv7_2_mbox_conf_flat",
        "conv7_2_mbox_priorbox",
        "conv8_2_mbox_loc_perm",
        "conv8_2_mbox_loc_flat",
        "conv8_2_mbox_conf_perm",
        "conv8_2_mbox_conf_flat",
        "conv8_2_mbox_priorbox",
        "conv9_2_mbox_loc_perm",
        "conv9_2_mbox_loc_flat",
        "conv9_2_mbox_conf_perm",
        "conv9_2_mbox_conf_flat",
        "conv9_2_mbox_priorbox",
        "mbox_loc",
        "mbox_conf",
        "mbox_priorbox",
        "mbox_conf_reshape",
        "mbox_conf_flatten",
        "detection_out"};

    TensorNet tensorNet;            // класс api TensorRT
    float** buffers;                // буфер с выделенной памятью для передачи в IExecutionContext.execute
    std::vector<float*> buf_layers; // контейнер с выделенной памятью для слоев

};


/*******************************/
// для работы с fast and opencv
bool loadImageRGBA(cv::Mat &img, float4 **cpu, float4 **gpu, int *width, int *height)
{

    if( !cpu || !gpu || !width || !height || img.empty() )
    {
        printf("loadImageRGBA - invalid parameter\n");
        return false;
    }

    const size_t imgSize = img.rows * img.cols * sizeof(float) * 4;

    // allocate buffer for the image
    if( !cudaAllocMapped((void**)cpu, (void**)gpu, imgSize) )
    {
        printf(LOG_CUDA "failed to allocated for image");
        return false;
    }

    float4* cpuPtr = *cpu;
    for (uint32_t y = 0; y < img.rows; y++) {
        for (uint32_t x = 0; x < img.cols; x++) {
            const float4 px = make_float4(float(img.at<cv::Vec3b>(y, x)[2]),
                                          float(img.at<cv::Vec3b>(y, x)[1]),
                                          float(img.at<cv::Vec3b>(y, x)[0]),
                                          float(255));
            cpuPtr[y*img.cols+x] = px;
        }
    }

    *width  = img.cols;
    *height = img.rows;
    return true;
}

// для работы с fast. Сохраняем сегментированный результат в cv::Mat
std::shared_ptr<cv::Mat> cpuSegment2cvMat( float4* cpu, int width, int height, float max_pixel ){

    auto out_img = std::make_shared<cv::Mat>(width, height, CV_8UC3);

    const float scale = 255.0f / max_pixel;
    //cv::Mat out_img(width, height, CV_8UC3);

    for( int y=0; y < height; y++ )
    {
        for( int x=0; x < width; x++ )
        {
            const float4 px = cpu[y * width + x];
            out_img->at<cv::Vec3b>(y,x)[2] = px.x*scale;
            out_img->at<cv::Vec3b>(y,x)[1] = px.y*scale;
            out_img->at<cv::Vec3b>(y,x)[0] = px.z*scale;
        }
    }

    return out_img;
}
/*************************************************************/

/*******************************/
// вычитание среднего
cudaError_t cudaPreImageNetMean( float4* input,
                                 size_t inputWidth,
                                 size_t inputHeight,
                                 float* output,
                                 size_t outputWidth,
                                 size_t outputHeight,
                                 const float3& mean_value );
/*************************************************************/

/*******************************/
// отрисовка bbox
template<typename T>
void draw_box(cv::Mat* img,
              std::vector<std::string> labels,
              T& detection_out_vec,
              int dims_data_x,
              int dims_data_y,
              const float scale_x,
              const float scale_y){

    if (img == nullptr){
        std::cerr<<"img for drawing boxes nullptr!"<<std::endl;
        return;
    }

    for(auto out:detection_out_vec){
        if (out->score < 0.85)
            continue;

        cv::rectangle(*img,
                      cv::Point((out->xmin*dims_data_x)/scale_x, (out->ymin*dims_data_y)/scale_y),
                      cv::Point((out->xmax*dims_data_x)/scale_x, (out->ymax*dims_data_y)/scale_y),
                      cv::Scalar(0,255,0),2);

        cv::putText(*img,labels.at(out->label),
                    cv::Point((out->xmin*dims_data_x)/scale_x, (out->ymin*dims_data_y)/scale_y),
                    1.2, 1.2, cv::Scalar(0,0,255), 2);
    }



}
/*************************************************************/

int main(int argc, char** argv)
{
    cv::VideoCapture cap("nvcamerasrc ! video/x-raw(memory:NVMM), width=(int)1280, height=(int)720,format=(string)I420, framerate=(fraction)24/1 ! nvvidconv flip-method=2 ! video/x-raw, format=(string)BGRx ! videoconvert ! video/x-raw, format=(string)BGR ! appsink"); //open the default camera
    if(!cap.isOpened())  // check if we succeeded
        return -1;

    NetworkModel network_model;


    cv::namedWindow("img",1);

    for(;;)
    {
        cv::Mat frame;
        cap >> frame;
        cv::flip(frame, frame, 0);
        cv::flip(frame, frame, 1);

        // преобразуем кадр для nn
        // инициализируем входное изображение
        float *img_cpu = nullptr;
        float *img_cuda = nullptr;
        int img_width = 0;
        int img_height = 0;

        // загружаем изображение в формате RGBA для перемещения картинки в память gpu
        if (!loadImageRGBA(frame, (float4 **) &img_cpu, (float4 **) &img_cuda, &img_width, &img_height)) {
            printf("failed to load image");
            return false;
        }

        float dims_data_w = network_model.get_dims_data().w();
        float dims_data_h = network_model.get_dims_data().h();

        // вычитание среднего параметрами ImageNet
        if( CUDA_FAILED(cudaPreImageNetMean((float4*)img_cuda,
                                                    (size_t)frame.cols,
                                                    (size_t)frame.rows,
                                                    network_model.get_buf_layer_data(),
                                                    dims_data_w,
                                                    dims_data_h,
                                                    make_float3(127.0f, 127.0f, 127.0f))) )
        {
            printf("cudaPreImageNetMean failed\n");
            return 0;
        }


        // детектирование
        network_model.image_inference();

        // задать вектор аттрибутов детектированного слоя
        network_model.set_detection_out();

        const float scale_x = dims_data_w / float(frame.cols);
        const float scale_y = dims_data_h / float(frame.rows);

        // отрисовка боксов
        draw_box(&frame,
                 network_model.labels,
                 network_model.detection_out,
                 (int)dims_data_w,
                 (int)dims_data_h,
                 scale_x,
                 scale_y);


        cv::imshow("img", frame);
        if(cv::waitKey(30) >= 0) break;

        // освобождаем память
        CUDA(cudaFreeHost(img_cpu));
        CUDA(cudaFreeHost(img_cuda));

    }

    return 0;
}
