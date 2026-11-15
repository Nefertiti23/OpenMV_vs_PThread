// Shared image-processing code used by all three programs.
// Each image is loaded, converted to grayscale, resized, blurred, and checksummed.
#pragma once
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <string>
#include <vector>
#include <algorithm>
#include <chrono>
#include <filesystem>
#include <cctype>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

static const int SZ = 500;   // every image is converted to SZ x SZ

inline double now_ms() {
    using namespace std::chrono;
    return duration<double, std::milli>(steady_clock::now().time_since_epoch()).count();
}


// Choose a spread-out subset when maxN is smaller than the number of files.
// maxN <= 0 means "use every image".
inline std::vector<std::string> list_pngs(const std::string& root, int maxN) {
    std::vector<std::string> all;
    namespace fs = std::filesystem;
    if (!fs::exists(root)) { fprintf(stderr, "Folder not found: %s\n", root.c_str()); return all; }
    for (auto& e : fs::recursive_directory_iterator(root)) {
        if (!e.is_regular_file()) continue;
        std::string ext = e.path().extension().string();
        for (auto& ch : ext) ch = (char)std::tolower((unsigned char)ch);
        if (ext == ".png") all.push_back(e.path().string());
    }
    std::sort(all.begin(), all.end());
    if (maxN <= 0 || (int)all.size() <= maxN) return all;
    std::vector<std::string> pick;
    pick.reserve(maxN);
    for (int i = 0; i < maxN; ++i) pick.push_back(all[(size_t)((long long)i * all.size() / maxN)]);
    return pick;
}

// Results and timing totals for a group of images.
struct Acc {
    uint64_t cs = 0;        // sum of per-image checksums (order independent)
    double load_ms = 0;     // time spent reading/decoding/grayscale/resizing (summed over images)
    double blur_ms = 0;     // time spent blurring (summed over images)
    int ok = 0, failed = 0;
};
inline void acc_add(Acc& a, const Acc& b) {
    a.cs += b.cs; a.load_ms += b.load_ms; a.blur_ms += b.blur_ms; a.ok += b.ok; a.failed += b.failed;
}

// Resize using area averaging.
inline void resize_box(const uint8_t* src, int w, int h, uint8_t* dst) {
    for (int y = 0; y < SZ; ++y) {
        long long y0 = (long long)y * h / SZ;
        long long y1 = std::max(y0 + 1, (long long)(y + 1) * h / SZ);
        for (int x = 0; x < SZ; ++x) {
            long long x0 = (long long)x * w / SZ;
            long long x1 = std::max(x0 + 1, (long long)(x + 1) * w / SZ);
            unsigned long long sum = 0;
            for (long long yy = y0; yy < y1; ++yy)
                for (long long xx = x0; xx < x1; ++xx) sum += src[(size_t)yy * w + xx];
            dst[y * SZ + x] = (uint8_t)(sum / (unsigned long long)((y1 - y0) * (x1 - x0)));
        }
    }
}

// Load an image and prepare its grayscale 500x500 version.
inline bool load_stage(const std::string& path, uint8_t* out500, Acc& acc) {
    double t0 = now_ms();
    int w, h, c;
    unsigned char* d = stbi_load(path.c_str(), &w, &h, &c, 1);   // 1 = convert to grayscale while decoding
    if (!d) { acc.failed++; return false; }
    resize_box(d, w, h, out500);
    stbi_image_free(d);
    acc.load_ms += now_ms() - t0;
    return true;
}

// Apply a 5x5 Gaussian blur with clamped borders.
inline void blur5x5(const uint8_t* in, uint8_t* out) {
    static const int K[5] = { 1, 4, 6, 4, 1 };
    for (int y = 0; y < SZ; ++y) {
        for (int x = 0; x < SZ; ++x) {
            int sum = 0;
            for (int ky = -2; ky <= 2; ++ky) {
                int yy = std::min(std::max(y + ky, 0), SZ - 1);
                for (int kx = -2; kx <= 2; ++kx) {
                    int xx = std::min(std::max(x + kx, 0), SZ - 1);
                    sum += in[yy * SZ + xx] * K[ky + 2] * K[kx + 2];
                }
            }
            out[y * SZ + x] = (uint8_t)(sum >> 8);
        }
    }
}

// Blur an image, update the checksum, and optionally keep the output.
inline void blur_stage(const uint8_t* in500, Acc& acc, uint8_t* keep_out = nullptr) {
    double t0 = now_ms();
    std::vector<uint8_t> out(SZ * SZ);
    blur5x5(in500, out.data());
    uint64_t cs = 0;
    for (size_t i = 0; i < out.size(); ++i) cs += (uint64_t)out[i] * (i % 251 + 1);
    acc.cs += cs;
    acc.ok++;
    if (keep_out) std::copy(out.begin(), out.end(), keep_out);
    acc.blur_ms += now_ms() - t0;
}

// Run both processing stages for one file.
inline void process_file(const std::string& path, Acc& acc) {
    std::vector<uint8_t> img(SZ * SZ);
    if (load_stage(path, img.data(), acc)) blur_stage(img.data(), acc);
}

// CSV output shared by all programs.
inline void print_header() {
    printf("impl,variant,threads,run,images,failed,total_ms,images_per_sec,load_cpu_ms,blur_cpu_ms,checksum\n");
    fflush(stdout);
}
inline void print_row(const char* impl, const char* variant, int threads, int run, const Acc& a, double total_ms) {
    printf("%s,%s,%d,%d,%d,%d,%.2f,%.2f,%.1f,%.1f,%llu\n", impl, variant, threads, run, a.ok, a.failed,
           total_ms, a.ok / (total_ms / 1000.0), a.load_ms, a.blur_ms, (unsigned long long)a.cs);
    fflush(stdout);
}