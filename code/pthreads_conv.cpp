// Pthreads version with three ways to divide the work:
//   simple   : fixed blocks and private totals
//   mutex    : shared work counter and shared totals protected by a mutex
//   combined : shared work counter with private totals merged at the end
// usage: pth.exe <dataset_folder> <threads> <runs> <simple|mutex|combined> <num_images>
//   num_images = 0 means "use every image"
#include <pthread.h>
#include "common.h"

struct Ctx {
    const std::vector<std::string>* files;
    int n, T;
    std::vector<Acc> priv;      // simple: one accumulator per thread
    pthread_mutex_t m;          // mutex / combined: protects `next` and `total`
    int next;
    Acc total;
};
struct Arg { Ctx* c; int id; };

// Fixed block per thread. No locking is needed while processing.
static void* worker_simple(void* p) {
    Arg* a = (Arg*)p; Ctx* c = a->c;
    int lo = (int)((long long)a->id * c->n / c->T);
    int hi = (int)((long long)(a->id + 1) * c->n / c->T);
    for (int i = lo; i < hi; ++i) process_file((*c->files)[i], c->priv[a->id]);
    return nullptr;
}

// Each thread takes the next available image.
static void* worker_mutex(void* p) {
    Ctx* c = ((Arg*)p)->c;
    for (;;) {
        pthread_mutex_lock(&c->m);
        int i = c->next++;
        pthread_mutex_unlock(&c->m);
        if (i >= c->n) break;
        Acc a;
        process_file((*c->files)[i], a);
        pthread_mutex_lock(&c->m);
        acc_add(c->total, a);
        pthread_mutex_unlock(&c->m);
    }
    return nullptr;
}

// Work is shared, but each thread keeps its own totals until the end.
static void* worker_combined(void* p) {
    Ctx* c = ((Arg*)p)->c;
    Acc mine;
    for (;;) {
        pthread_mutex_lock(&c->m);
        int i = c->next++;
        pthread_mutex_unlock(&c->m);
        if (i >= c->n) break;
        process_file((*c->files)[i], mine);
    }
    pthread_mutex_lock(&c->m);
    acc_add(c->total, mine);
    pthread_mutex_unlock(&c->m);
    return nullptr;
}

static void* empty_fn(void*) { return nullptr; }

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
    int n = (int)files.size();
    fprintf(stderr, "found %d images to process\n", n);
    if (n == 0) return 1;

    print_header();
    for (int r = 1; r <= runs; ++r) {
        Ctx c; c.files = &files; c.n = n; c.T = T; c.next = 0; c.priv.assign(T, Acc());
        pthread_mutex_init(&c.m, nullptr);
        std::vector<pthread_t> th(T);
        std::vector<Arg> args(T);
        void* (*fn)(void*) = variant == "simple" ? worker_simple : variant == "mutex" ? worker_mutex : worker_combined;

        double t0 = now_ms();
        for (int t = 0; t < T; ++t) { args[t].c = &c; args[t].id = t; pthread_create(&th[t], nullptr, fn, &args[t]); }
        for (int t = 0; t < T; ++t) pthread_join(th[t], nullptr);
        double ms = now_ms() - t0;

        Acc result = c.total;
        if (variant == "simple") for (int t = 0; t < T; ++t) acc_add(result, c.priv[t]);
        print_row("pthreads", variant.c_str(), T, r, result, ms);
        pthread_mutex_destroy(&c.m);
    }

    // Measure the cost of creating and joining threads.
    const int REP = 1000;
    std::vector<pthread_t> th(T);
    double t0 = now_ms();
    for (int k = 0; k < REP; ++k) {
        for (int t = 0; t < T; ++t) pthread_create(&th[t], nullptr, empty_fn, nullptr);
        for (int t = 0; t < T; ++t) pthread_join(th[t], nullptr);
    }
    double per_us = (now_ms() - t0) * 1000.0 / REP;
    fprintf(stderr, "OVERHEAD pthreads: %.2f microseconds to create+join %d threads\n", per_us, T);
    return 0;
}