#include <opencv2/opencv.hpp>
#include <omp.h>
#include <filesystem>
#include <string>
#include <iostream>
#include <chrono>

// this program is WITH mutex locks

void blur_the_image(const std::string& input_path, const std::string& output_folder) {
    cv::Mat img = cv::imread(input_path, cv::IMREAD_GRAYSCALE);
    if (img.empty()) {
        std::cerr << "Failed to read image: " << input_path << "\n";
        return;
    }

    cv::Mat blur_img;
    // {7, 7} kernel, with each cell containing the value 1/25
    cv::GaussianBlur(img, blur_img, cv::Size(5, 5), 0);;

    std::string output_path = output_folder + "/" + std::filesystem::path(input_path).filename().string();

    if (!cv::imwrite(output_path, blur_img)) {
        std::cerr << "Failed to save image: " << output_path << "\n";
    }
}

int main() {
    std::string input_folder = "../big_data/train";
    std::string output_folder = "../output_big_data";

    std::filesystem::create_directories(output_folder);

    std::vector<std::string> image_files;
    for (const auto& entry : std::filesystem::directory_iterator(input_folder)) {
        if (entry.is_regular_file()) {
            image_files.emplace_back(entry.path().string());
        }
    }

    std::cout << "Found " << image_files.size() << " images. Processing...\n";
    auto start_time = std::chrono::high_resolution_clock::now();

    // Parallel processing using OpenMP
    int next = 0;
    #pragma omp parallel
    {
        int i;
        while (true) {
            // using locks
            #pragma omp atomic capture
            {
                i = next;
                next++;
            }

            if (i >= image_files.size()) break;

            // Process image outside the lock/atomic block
            blur_the_image(image_files[i], output_folder);
        }
    }

    auto end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> elapsed = end_time - start_time;
    std::cout << "Execution Runtime: " << elapsed.count() << " ms" << std::endl;

    std::cout << "Processing complete." << std::endl;
    return 0;
}

/*
clang++ -std=c++17 -O3 processor.cpp -o processor \
  -Xpreprocessor -fopenmp \
  -I/opt/homebrew/opt/libomp/include \
  -L/opt/homebrew/opt/libomp/lib -lomp \
  -I/opt/homebrew/opt/opencv/include/opencv5 \
  -L/opt/homebrew/opt/opencv/lib \
  -lopencv_core -lopencv_imgcodecs -lopencv_imgproc
*/
