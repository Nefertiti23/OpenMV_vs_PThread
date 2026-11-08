// pthreads_conv.cpp - POSIX threads version (dynamic work queue: mutex-protected counter)
// usage: pth.exe <data_dir> <threads> <runs> <passes> <max_images>
#include <pthread.h>
#include "common.h"
#include <chrono>
#include <iostream>

struct Shared {
    const std::vector<Img>* imgs;
    std::vector<uint64_t>* res;
    int next;
    int n;
    int passes;
    pthread_mutex_t m;
};

static void* worker(void* arg) {
    Shared* s = (Shared*)arg;
    for (;;) {
        pthread_mutex_lock(&s->m);
        int i = s->next++;
        pthread_mutex_unlock(&s->m);
        if (i >= s->n) break;
        (*s->res)[i] = process_cs((*s->imgs)[i], s->passes);
    }
    return nullptr;
}

static void* empty_fn(void*) { return nullptr; }

int main(int argc, char** argv) {
    std::string dir = argc > 1 ? argv[1] : "../big_data/train/";
    int T      = argc > 2 ? atoi(argv[2]) : 4;
    int runs   = argc > 3 ? atoi(argv[3]) : 1;
    int passes = argc > 4 ? atoi(argv[4]) : 1;
    int maxN   = argc > 5 ? atoi(argv[5]) : 67000;

    std::vector<Img> imgs = load_all(dir, maxN);
    int n = (int)imgs.size();
    if (n == 0) return 1;
    std::vector<uint64_t> res(n);

    // printf("impl,schedule,threads,run,images,passes,compute_ms,images_per_sec,checksum\n");
    auto start_time = std::chrono::high_resolution_clock::now();
    for (int r = 0; r < runs; ++r) {
        Shared s; s.imgs = &imgs; s.res = &res; s.next = 0; s.n = n; s.passes = passes;
        pthread_mutex_init(&s.m, nullptr);
        std::vector<pthread_t> th(T);
        double t0 = now_ms();
        for (int t = 0; t < T; ++t) pthread_create(&th[t], nullptr, worker, &s);
        for (int t = 0; t < T; ++t) pthread_join(th[t], nullptr);
        double ms = now_ms() - t0;
        pthread_mutex_destroy(&s.m);
        uint64_t total = 0;
        for (int i = 0; i < n; ++i) total += res[i];
        // printf("pthreads,dynamic,%d,%d,%d,%d,%.2f,%.2f,%llu\n", T, r + 1, n, passes, ms,
        //        n / (ms / 1000.0), (unsigned long long)total);
    }

    auto end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> elapsed = end_time - start_time;
    std::cout << "Total Workload Execution Runtime (" << runs << " runs): " << elapsed.count() << " ms\n";

    // thread-creation overhead: create+join T empty threads, repeated 1000 times
    const int REP = 1000;
    std::vector<pthread_t> th(T);
    double t0 = now_ms();
    for (int k = 0; k < REP; ++k) {
        for (int t = 0; t < T; ++t) pthread_create(&th[t], nullptr, empty_fn, nullptr);
        for (int t = 0; t < T; ++t) pthread_join(th[t], nullptr);
    }
    double per = (now_ms() - t0) * 1000.0 / REP;
    fprintf(stderr, "OVERHEAD,pthreads,%d,%.2f us per create+join of %d threads\n", T, per, T);
    printf("overhead,pthreads,%d,0,0,0,%.5f,0,0\n", T, per / 1000.0); // value in ms (column compute_ms)

    return 0;
}
