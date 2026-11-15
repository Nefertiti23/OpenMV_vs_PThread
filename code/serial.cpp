// Serial version used as the baseline for comparison.
// usage: serial.exe <dataset_folder> <num_images> <runs>
//   num_images = 0 means "use every image"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"
#include "common.h"

int main(int argc, char** argv) {
    std::string dir = argc > 1 ? argv[1] : "C:\\Users\\user\\Desktop\\OpenMV_vs_PThread\\big_data\\train";
    int maxN = argc > 2 ? atoi(argv[2]) : 1000;
    int runs = argc > 3 ? atoi(argv[3]) : 3;

    std::vector<std::string> files = list_pngs(dir, maxN);
    int n = (int)files.size();
    fprintf(stderr, "found %d images to process\n", n);
    if (n == 0) return 1;

    // Save one sample before and after the blur.
    {
        Acc tmp; std::vector<uint8_t> in(SZ * SZ), out(SZ * SZ);
        if (load_stage(files[0], in.data(), tmp)) {
            blur_stage(in.data(), tmp, out.data());
            stbi_write_png("sample_in.png", SZ, SZ, 1, in.data(), SZ);
            stbi_write_png("sample_out.png", SZ, SZ, 1, out.data(), SZ);
        }
    }

    print_header();
    for (int r = 1; r <= runs; ++r) {
        Acc total;
        double t0 = now_ms();
        for (int i = 0; i < n; ++i) process_file(files[i], total);
        print_row("serial", "serial", 1, r, total, now_ms() - t0);
    }
    return 0;
}