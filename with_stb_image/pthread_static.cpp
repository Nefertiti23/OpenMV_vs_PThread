// pthreads_nomutex.cpp - POSIX threads version WITHOUT a mutex (static partition)
// Each thread gets its own fixed block of images, so no shared counter and no lock.
// usage: pth_nomutex.exe <data_dir> <threads> <runs> <passes> <max_images>
#include <pthread.h>
#include "common.h"
#include <chrono>
#include <iostream>

struct Task {
    const std::vector<Img>* imgs;
    std::vector<uint64_t>* res;
    int begin, end;
    int passes;
};

static void* worker(void* arg) {
    Task* t = (Task*)arg;
    for (int i = t->begin; i < t->end; ++i)
        (*t->res)[i] = process_cs((*t->imgs)[i], t->passes);
    return nullptr;
}

static void* empty_fn(void*) { return nullptr; }

int main(int argc, char** argv) {
    std::string dir = argc > 1 ? argv[1] : "../big_data/train/";
    int T      = argc > 2 ? atoi(argv[2]) : 4;
    int runs   = argc > 3 ? atoi(argv[3]) : 1;
    int passes = argc > 4 ? atoi(argv[4]) : 1;
    int maxN   = argc > 5 ? atoi(argv[5]) : 67400;

    std::vector<Img> imgs = load_all(dir, maxN);
    int n = (int)imgs.size();
    if (n == 0) return 1;
    std::vector<uint64_t> res(n);
    uint64_t total = 0;

    auto start_time = std::chrono::high_resolution_clock::now();
    for (int r = 0; r < runs; ++r) {
        std::vector<Task> tasks(T);
        int chunk = (n + T - 1) / T;                 // images per thread (rounded up)
        for (int t = 0; t < T; ++t) {
            tasks[t].imgs = &imgs; tasks[t].res = &res; tasks[t].passes = passes;
            tasks[t].begin = std::min(n, t * chunk);
            tasks[t].end   = std::min(n, tasks[t].begin + chunk);
        }
        std::vector<pthread_t> th(T);
        for (int t = 0; t < T; ++t) pthread_create(&th[t], nullptr, worker, &tasks[t]);
        for (int t = 0; t < T; ++t) pthread_join(th[t], nullptr);
        total = 0;
        for (int i = 0; i < n; ++i) total += res[i];
    }
    auto end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> elapsed = end_time - start_time;
    std::cout << "Total Workload Execution Runtime (" << runs << " runs): " << elapsed.count() << " ms\n";
    std::cout << "Checksum: " << total << "\n";

    // thread-creation overhead: create+join T empty threads, repeated 1000 times
    const int REP = 1000;
    std::vector<pthread_t> th(T);
    double t0 = now_ms();
    for (int k = 0; k < REP; ++k) {
        for (int t = 0; t < T; ++t) pthread_create(&th[t], nullptr, empty_fn, nullptr);
        for (int t = 0; t < T; ++t) pthread_join(th[t], nullptr);
    }
    double per = (now_ms() - t0) * 1000.0 / REP;
    fprintf(stderr, "OVERHEAD,pthreads_nomutex,%d,%.2f us per create+join of %d threads\n", T, per, T);
    printf("overhead,pthreads_nomutex,%d,0,0,0,%.5f,0,0\n", T, per / 1000.0);
    return 0;
}
