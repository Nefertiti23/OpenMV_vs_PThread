#pragma once

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <stdexcept>
#include <string>
#include <vector>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

struct BenchmarkImage {
    static constexpr int width = 500;
    static constexpr int height = 500;
    std::vector<std::uint8_t> pixels;
};

inline std::vector<BenchmarkImage> load_images(const std::filesystem::path& directory,
                                               int max_images) {
    if (!std::filesystem::is_directory(directory)) {
        throw std::runtime_error("Input directory does not exist: " + directory.string());
    }

    std::vector<std::filesystem::path> files;
    for (const auto& entry : std::filesystem::directory_iterator(directory)) {
        if (entry.is_regular_file() && entry.path().extension() == ".png") {
            files.push_back(entry.path());
        }
    }
    std::sort(files.begin(), files.end());
    if (max_images > 0 && static_cast<int>(files.size()) > max_images) {
        files.resize(static_cast<std::size_t>(max_images));
    }

    std::vector<BenchmarkImage> images;
    images.reserve(files.size());
    for (const auto& file : files) {
        int width = 0;
        int height = 0;
        int channels = 0;
        unsigned char* pixels = stbi_load(file.string().c_str(), &width, &height, &channels, 1);
        if (pixels == nullptr) {
            throw std::runtime_error("Failed to load image: " + file.string());
        }
        BenchmarkImage image;
        image.pixels.resize(static_cast<std::size_t>(BenchmarkImage::width) *
                            BenchmarkImage::height);
        for (int y = 0; y < BenchmarkImage::height; ++y) {
            const int source_y = y * height / BenchmarkImage::height;
            for (int x = 0; x < BenchmarkImage::width; ++x) {
                const int source_x = x * width / BenchmarkImage::width;
                image.pixels[static_cast<std::size_t>(y) * BenchmarkImage::width + x] =
                    pixels[static_cast<std::size_t>(source_y) * width + source_x];
            }
        }
        stbi_image_free(pixels);
        images.push_back(std::move(image));
    }

    if (images.empty()) {
        throw std::runtime_error("No PNG images found in: " + directory.string());
    }
    return images;
}

inline std::uint64_t process_image(const BenchmarkImage& image, int passes) {
    static constexpr int kernel[5] = {1, 4, 6, 4, 1};
    const int width = BenchmarkImage::width;
    const int height = BenchmarkImage::height;
    std::vector<std::uint8_t> current = image.pixels;
    std::vector<std::uint8_t> next(current.size());

    for (int pass = 0; pass < passes; ++pass) {
        for (int y = 0; y < height; ++y) {
            for (int x = 0; x < width; ++x) {
                int sum = 0;
                for (int ky = -2; ky <= 2; ++ky) {
                    const int sample_y = std::clamp(y + ky, 0, height - 1);
                    for (int kx = -2; kx <= 2; ++kx) {
                        const int sample_x = std::clamp(x + kx, 0, width - 1);
                        sum += current[static_cast<std::size_t>(sample_y) * width + sample_x] *
                               kernel[ky + 2] * kernel[kx + 2];
                    }
                }
                next[static_cast<std::size_t>(y) * width + x] =
                    static_cast<std::uint8_t>(sum >> 8);
            }
        }
        current.swap(next);
    }

    std::uint64_t checksum = 0;
    for (std::size_t i = 0; i < current.size(); ++i) {
        checksum += static_cast<std::uint64_t>(current[i]) * (i % 251 + 1);
    }
    return checksum;
}

inline void validate_arguments(int threads, int runs, int passes) {
    if (threads <= 0 || runs <= 0 || passes <= 0) {
        throw std::invalid_argument("threads, runs, and passes must all be positive");
    }
}
