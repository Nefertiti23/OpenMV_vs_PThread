// OpenMP version with the same three work-sharing methods as the Pthreads version:
//   simple   : parallel loop with reduction
//   mutex    : dynamic scheduling with a shared locked total
//   combined : dynamic scheduling with private totals merged at the end
//
// The thread count is chosen by OpenMP or the OMP_NUM_THREADS environment variable.
//
// usage: omp.exe <dataset_folder> <runs> <simple|mutex|combined> <num_images>
//   num_images = 0 means "use every image"
#include <omp.h>
#include "common.h"

int main(int argc, char** argv) {
    std::string dir     = argc > 1 ? argv[1] : "C:\\Users\\user\\Desktop\\OpenMV_vs_PThread\\big_data\\train";
    int runs            = argc > 2 ? atoi(argv[2]) : 3;
    std::string variant = argc > 3 ? argv[3] : "simple";
    int maxN            = argc > 4 ? atoi(argv[4]) : 1000;
    if (variant != "simple" && variant != "mutex" && variant != "combined") {
        fprintf(stderr, "variant must be simple, mutex or combined\n"); return 1;
    }

    std::vector<std::string> files = list_pngs(dir, maxN);
    const int n = (int)files.size();
    fprintf(stderr, "found %d images to process\n", n);
    if (n == 0) return 1;

    // Record the number of threads selected by OpenMP.
    int used = 1;
    #pragma omp parallel
    {
        #pragma omp single
        used = omp_get_num_threads();
    }
    fprintf(stderr, "OpenMP chose %d threads\n", used);

    print_header();
    for (int r = 1; r <= runs; ++r) {
        Acc total;
        double t0 = now_ms();

        if (variant == "simple") {
            unsigned long long cs = 0; double lo = 0, bl = 0; int ok = 0, fail = 0;
            #pragma omp parallel for schedule(static) reduction(+:cs,lo,bl,ok,fail)
            for (int i = 0; i < n; ++i) {
                Acc a;
                process_file(files[i], a);
                cs += a.cs; lo += a.load_ms; bl += a.blur_ms; ok += a.ok; fail += a.failed;
            }
            total.cs = cs; total.load_ms = lo; total.blur_ms = bl; total.ok = ok; total.failed = fail;
        }
        else if (variant == "mutex") {
            omp_lock_t lock; omp_init_lock(&lock);
            #pragma omp parallel for schedule(dynamic, 1)
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
            #pragma omp parallel
            {
                Acc mine;
                #pragma omp for schedule(dynamic, 1) nowait
                for (int i = 0; i < n; ++i) process_file(files[i], mine);
                omp_set_lock(&lock);
                acc_add(total, mine);
                omp_unset_lock(&lock);
            }
            omp_destroy_lock(&lock);
        }

        double ms = now_ms() - t0;
        print_row("openmp", variant.c_str(), used, r, total, ms);
    }

    // Measure the cost of starting empty parallel regions.
    const int REP = 1000;
    volatile int sink = 0;
    double t0 = now_ms();
    for (int k = 0; k < REP; ++k) {
        #pragma omp parallel
        {
            sink = 1;
        }
    }
    double per_us = (now_ms() - t0) * 1000.0 / REP;
    fprintf(stderr, "OVERHEAD openmp: %.2f microseconds per parallel region of %d threads\n", per_us, used);
    return 0;
}