// common.h - shared code: image loading, 5x5 Gaussian convolution, checksum, timer
#pragma once
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <string>
#include <vector>
#include <algorithm>
#include <chrono>
#include <filesystem>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

struct Img {
    int w = 0, h = 0;
    std::vector<uint8_t> px;   // grayscale, w*h bytes
};

inline double now_ms() {
    using namespace std::chrono;
    return duration<double, std::milli>(steady_clock::now().time_since_epoch()).count();
}

// Load up to maxN PNG files from dir as 8-bit grayscale (done once, NOT timed as compute)
inline std::vector<Img> load_all(const std::string& dir, int maxN) {
    std::vector<std::string> files;
    for (auto& e : std::filesystem::directory_iterator(dir))
        if (e.path().extension() == ".png") files.push_back(e.path().string());
    std::sort(files.begin(), files.end());
    if (maxN > 0 && (int)files.size() > maxN) files.resize(maxN);
    std::vector<Img> imgs;
    imgs.reserve(files.size());
    for (auto& f : files) {
        int w, h, c;
        unsigned char* d = stbi_load(f.c_str(), &w, &h, &c, 1);
        if (!d) { fprintf(stderr, "failed to load %s\n", f.c_str()); continue; }
        Img im; im.w = w; im.h = h; im.px.assign(d, d + (size_t)w * h);
        stbi_image_free(d);
        imgs.push_back(std::move(im));
    }
    return imgs;
}

// 5x5 Gaussian blur (kernel = [1 4 6 4 1] x [1 4 6 4 1], sum 256), clamped borders.
// Applies it `passes` times. Writes the result into `out`, returns a checksum of it.
inline uint64_t process(const Img& im, int passes, std::vector<uint8_t>& out) {
    static const int K[5] = { 1, 4, 6, 4, 1 };
    const int w = im.w, h = im.h;
    std::vector<uint8_t> a(im.px), b((size_t)w * h);
    for (int p = 0; p < passes; ++p) {
        for (int y = 0; y < h; ++y) {
            for (int x = 0; x < w; ++x) {
                int sum = 0;
                for (int ky = -2; ky <= 2; ++ky) {
                    int yy = std::min(std::max(y + ky, 0), h - 1);
                    for (int kx = -2; kx <= 2; ++kx) {
                        int xx = std::min(std::max(x + kx, 0), w - 1);
                        sum += a[(size_t)yy * w + xx] * K[ky + 2] * K[kx + 2];
                    }
                }
                b[(size_t)y * w + x] = (uint8_t)(sum >> 8);
            }
        }
        a.swap(b);
    }
    out.swap(a);
    uint64_t cs = 0;
    for (size_t i = 0; i < out.size(); ++i) cs += (uint64_t)out[i] * (i % 251 + 1);
    return cs;
}

inline uint64_t process_cs(const Img& im, int passes) {
    std::vector<uint8_t> tmp;
    return process(im, passes, tmp);
}
