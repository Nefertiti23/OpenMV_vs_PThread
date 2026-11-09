// openmp_conv.cpp - OpenMP, three versions (same three as the Pthreads program):
//   simple   : parallel for + reduction (no explicit locks)
//   mutex    : dynamic parallel for, shared totals protected by an OpenMP lock (omp_lock_t = mutex)
//   combined : simple + mutex together: dynamic scheduling, private totals per thread,
//              merged once at the end under an OpenMP lock
// usage: omp.exe <dataset_folder> <threads> <runs> <simple|mutex|combined> <num_images>
//   num_images = 0 means "use every image"
#include <omp.h>
#include "common.h"

int main(int argc, char** argv) {
    std::string dir     = argc > 1 ? argv[1] : "C:\\Users\\user\\Desktop\\OpenMV_vs_PThread\\big_data\\train";
    int T               = argc > 2 ? atoi(argv[2]) : 4;
    int runs            = argc > 3 ? atoi(argv[3]) : 3;
    std::string variant = argc > 4 ? argv[4] : "simple";
    int maxN            = argc > 5 ? atoi(argv[5]) : 1000;
    if (variant != "simple" && variant != "mutex" && variant != "combined") {
        fprintf(stderr, "variant must be simple, mutex or combined\n"); return 1;
    }

    std::vector<std::string> files = list_pngs(dir, maxN);
    const int n = (int)files.size();
    fprintf(stderr, "found %d images to process\n", n);
    if (n == 0) return 1;

    omp_set_dynamic(0);   // use exactly T threads

    print_header();
    for (int r = 1; r <= runs; ++r) {
        Acc total;
        double t0 = now_ms();

        if (variant == "simple") {
            unsigned long long cs = 0; double lo = 0, bl = 0; int ok = 0, fail = 0;
            #pragma omp parallel for num_threads(T) schedule(static) reduction(+:cs,lo,bl,ok,fail)
            for (int i = 0; i < n; ++i) {
                Acc a;
                process_file(files[i], a);
                cs += a.cs; lo += a.load_ms; bl += a.blur_ms; ok += a.ok; fail += a.failed;
            }
            total.cs = cs; total.load_ms = lo; total.blur_ms = bl; total.ok = ok; total.failed = fail;
        }
        else if (variant == "mutex") {
            omp_lock_t lock; omp_init_lock(&lock);
            #pragma omp parallel for num_threads(T) schedule(dynamic, 1)
            for (int i = 0; i < n; ++i) {
                Acc a;
                process_file(files[i], a);
                omp_set_lock(&lock);
                acc_add(total, a);
                omp_unset_lock(&lock);
            }
            omp_destroy_lock(&lock);
        }
        else { // combined = simple + mutex
            omp_lock_t lock; omp_init_lock(&lock);
            #pragma omp parallel num_threads(T)
            {
                Acc mine;                                  // private totals (simple part)
                #pragma omp for schedule(dynamic, 1) nowait
                for (int i = 0; i < n; ++i) process_file(files[i], mine);
                omp_set_lock(&lock);                       // one locked merge per thread (mutex part)
                acc_add(total, mine);
                omp_unset_lock(&lock);
            }
            omp_destroy_lock(&lock);
        }

        double ms = now_ms() - t0;
        print_row("openmp", variant.c_str(), T, r, total, ms);
    }

    // parallel-region startup overhead: an empty parallel region with T threads, repeated 1000 times
    const int REP = 1000;
    volatile int sink = 0;
    double t0 = now_ms();
    for (int k = 0; k < REP; ++k) {
        #pragma omp parallel num_threads(T)
        {
            sink = 1;
        }
    }
    double per_us = (now_ms() - t0) * 1000.0 / REP;
    fprintf(stderr, "OVERHEAD openmp: %.2f microseconds per parallel region of %d threads\n", per_us, T);
    return 0;
}