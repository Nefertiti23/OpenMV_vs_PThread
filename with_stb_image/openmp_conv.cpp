#include <omp.h>
#include <cstring>
#include "common.h"
#include <iostream>

// openmp_conv.cpp - OpenMP version
// usage: omp.exe <data_dir> <threads> <runs> <passes> <schedule: simple|muyex|combined> <max_images>
// NOTE: MSVC supports OpenMP 2.0 only, so the loop variable must be a signed int.

int main(int argc, char** argv) {
    std::string dir = argc > 1 ? argv[1] : "../big_data/train";
    int T      = argc > 2 ? atoi(argv[2]) : 4;
    int runs   = argc > 3 ? atoi(argv[3]) : 1;
    int passes = argc > 4 ? atoi(argv[4]) : 1;
    std::string sched = argc > 5 ? argv[5] : "combined";
    int maxN   = argc > 6 ? atoi(argv[6]) : 67400;

    std::vector<Img> imgs = load_all(dir, maxN);
    int n = (int)imgs.size();
    if (n == 0) return 1;
    std::vector<uint64_t> res(n);

    printf("impl,schedule,threads,run,images,passes,compute_ms,images_per_sec,checksum\n");
    auto start_time = std::chrono::high_resolution_clock::now();
    for (int r = 0; r < runs; ++r) {
        double t0 = now_ms();

        if (sched == "simple") {
            #pragma omp parallel for num_threads(T) schedule(static)
            for (int i = 0; i < n; ++i) {
                res[i] = process_cs(imgs[i], passes);
            }

        } else if (sched == "mutex") {
            int next = 0;
            #pragma omp parallel num_threads(T)
            {
                int i;
                while (true) {
                    #pragma omp atomic capture
                    {
                        i = next;
                        next++;
                    }
                    if (i >= n) break;
                    res[i] = process_cs(imgs[i], passes);
                }
            }

        } else if (sched == "combined") {
            uint64_t running_checksum = 0;

            #pragma omp parallel for num_threads(T) schedule(static)
            for (int i = 0; i < n; ++i) {
                res[i] = process_cs(imgs[i], passes);

                // Mutex constraint injected inside the simple loop
                #pragma omp critical
                {
                    running_checksum += res[i];
                }
            }
        }

        double ms = now_ms() - t0;
        uint64_t total = 0;
        for (int i = 0; i < n; ++i) total += res[i];
        printf("openmp,%s,%d,%d,%d,%d,%.2f,%.2f,%llu\n", sched.c_str(), T, r + 1, n, passes, ms,
               n / (ms / 1000.0), (unsigned long long)total);
    }

    auto end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> elapsed = end_time - start_time;
    std::cout << "Total Workload Execution Runtime (" << runs << " runs): " << elapsed.count() << " ms\n";

    // parallel-region startup overhead: empty region with T threads, repeated 1000 times
    const int REP = 1000;
    volatile int sink = 0;
    double t0 = now_ms();
    for (int k = 0; k < REP; ++k) {
        #pragma omp parallel num_threads(T)
        {
            sink = 1;
        }
    }
    double per = (now_ms() - t0) * 1000.0 / REP;
    fprintf(stderr, "OVERHEAD,openmp,%d,%.2f us per parallel region of %d threads\n", T, per, T);
    printf("overhead,openmp,%d,0,0,0,%.5f,0,0\n", T, per / 1000.0); // value in ms (column compute_ms)
    return 0;
}
