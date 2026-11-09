#include "benchmark_common.h"

#include <chrono>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <pthread.h>
#include <string>
#include <vector>

struct PthreadState {
    const std::vector<BenchmarkImage>* images;
    std::vector<std::uint64_t>* results;
    int next_image = 0;
    int passes = 1;
    pthread_mutex_t mutex;
};

void* pthread_worker(void* argument) {
    auto* state = static_cast<PthreadState*>(argument);
    for (;;) {
        pthread_mutex_lock(&state->mutex);
        const int index = state->next_image++;
        pthread_mutex_unlock(&state->mutex);
        if (index >= static_cast<int>(state->images->size())) {
            return nullptr;
        }
        (*state->results)[index] = process_image((*state->images)[index], state->passes);
    }
}

void* empty_worker(void*) {
    return nullptr;
}

std::uint64_t sum_results(const std::vector<std::uint64_t>& results) {
    std::uint64_t checksum = 0;
    for (const auto result : results) {
        checksum += result;
    }
    return checksum;
}

double pthread_overhead_us(int threads) {
    constexpr int repetitions = 100;
    std::vector<pthread_t> handles(static_cast<std::size_t>(threads));
    const auto start = std::chrono::steady_clock::now();
    for (int repetition = 0; repetition < repetitions; ++repetition) {
        for (int i = 0; i < threads; ++i) {
            if (pthread_create(&handles[i], nullptr, empty_worker, nullptr) != 0) {
                throw std::runtime_error("pthread_create failed during overhead measurement");
            }
        }
        for (int i = 0; i < threads; ++i) {
            if (pthread_join(handles[i], nullptr) != 0) {
                throw std::runtime_error("pthread_join failed during overhead measurement");
            }
        }
    }
    const auto elapsed =
        std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - start);
    return elapsed.count() / repetitions;
}

double openmp_overhead_us(int threads) {
    constexpr int repetitions = 1000;
    omp_set_num_threads(threads);
    const double start = omp_get_wtime();
    for (int repetition = 0; repetition < repetitions; ++repetition) {
#pragma omp parallel
        {
        }
    }
    return (omp_get_wtime() - start) * 1'000'000.0 / repetitions;
}

double run_pthread(const std::vector<BenchmarkImage>& images, int threads, int passes,
                   std::vector<std::uint64_t>& results) {
    PthreadState state{&images, &results, 0, passes, {}};
    if (pthread_mutex_init(&state.mutex, nullptr) != 0) {
        throw std::runtime_error("pthread_mutex_init failed");
    }
    std::vector<pthread_t> handles(static_cast<std::size_t>(threads));
    const auto start = std::chrono::steady_clock::now();
    for (int i = 0; i < threads; ++i) {
        if (pthread_create(&handles[i], nullptr, pthread_worker, &state) != 0) {
            pthread_mutex_destroy(&state.mutex);
            throw std::runtime_error("pthread_create failed during workload");
        }
    }
    for (int i = 0; i < threads; ++i) {
        if (pthread_join(handles[i], nullptr) != 0) {
            pthread_mutex_destroy(&state.mutex);
            throw std::runtime_error("pthread_join failed during workload");
        }
    }
    pthread_mutex_destroy(&state.mutex);
    return std::chrono::duration<double, std::milli>(
               std::chrono::steady_clock::now() - start)
        .count();
}

double run_openmp(const std::vector<BenchmarkImage>& images, int threads, int passes,
                  std::vector<std::uint64_t>& results) {
    omp_set_dynamic(0);
    omp_set_num_threads(threads);
    const double start = omp_get_wtime();
#pragma omp parallel for schedule(static)
    for (int i = 0; i < static_cast<int>(images.size()); ++i) {
        results[i] = process_image(images[i], passes);
    }
    return (omp_get_wtime() - start) * 1000.0;
}

void write_row(std::ostream& output, const std::string& implementation, int threads, int run,
               std::size_t image_count, int passes, double milliseconds, double baseline_ms,
               double overhead_us, std::uint64_t checksum) {
    const double throughput = image_count / (milliseconds / 1000.0);
    const double speedup = baseline_ms / milliseconds;
    output << implementation << ',' << threads << ',' << run << ',' << image_count << ','
           << passes << ',' << std::fixed << std::setprecision(3) << milliseconds << ','
           << throughput << ',' << speedup << ',' << overhead_us << ',' << checksum << '\n';
}

int main(int argc, char** argv) {
    try {
        const std::filesystem::path directory = argc > 1 ? argv[1] : "big_data/train";
        const int max_images = argc > 2 ? std::atoi(argv[2]) : 67000;
        const int runs = argc > 3 ? std::atoi(argv[3]) : 3;
        const int passes = argc > 4 ? std::atoi(argv[4]) : 1;
        validate_arguments(1, runs, passes);

        const auto images = load_images(directory, max_images);
        const std::vector<int> thread_counts = {1, 2, 4, 8};
        std::ofstream csv("combined_results.csv");
        if (!csv) {
            throw std::runtime_error("Could not create combined_results.csv");
        }

        csv << "implementation,threads,run,images,passes,compute_ms,images_per_sec,"
               "speedup_vs_pthread_1_thread,overhead_us,checksum\n";
        std::vector<std::uint64_t> results(images.size());
        double pthread_one_thread_baseline = 0.0;

        for (const int threads : thread_counts) {
            for (int run = 1; run <= runs; ++run) {
                const double milliseconds =
                    run_pthread(images, threads, passes, results);
                const auto checksum = sum_results(results);
                if (threads == 1 && pthread_one_thread_baseline == 0.0) {
                    pthread_one_thread_baseline = milliseconds;
                }
                write_row(csv, "pthread", threads, run, images.size(), passes, milliseconds,
                          pthread_one_thread_baseline, pthread_overhead_us(threads), checksum);
            }
        }

        for (const int threads : thread_counts) {
            for (int run = 1; run <= runs; ++run) {
                const double milliseconds =
                    run_openmp(images, threads, passes, results);
                const auto checksum = sum_results(results);
                write_row(csv, "openmp", threads, run, images.size(), passes, milliseconds,
                          pthread_one_thread_baseline, openmp_overhead_us(threads), checksum);
            }
        }

        std::cout << "Wrote combined_results.csv for " << images.size()
                  << " images. Compare matching checksums before using timing results.\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "error: " << error.what() << '\n';
        return 1;
    }
}
