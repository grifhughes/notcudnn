#include "layer.h"

struct layer *layer_create(int ro, int co, int bs)
{
        struct layer *l;

        cudaMallocManaged(&l, sizeof(struct layer));
        if(!l) {
                fprintf(stderr, "alloc failed, exiting...");
                exit(-1);
        }
        cudaMalloc((void **)&(l->w), sizeof(float) * ro * co);
        if(!l->w) {
                fprintf(stderr, "alloc failed, exiting...");
                exit(-1);
        }
        cudaMalloc((void **)&(l->b), sizeof(float) * ro * bs);
        if(!l->b) {
                fprintf(stderr, "alloc failed, exiting...");
                exit(-1);
        }
        cudaMalloc((void **)&(l->wi), sizeof(float) * ro * bs);
        if(!l->wi) {
                fprintf(stderr, "alloc failed, exiting...");
                exit(-1);
        }
        cudaMalloc((void **)&(l->a), sizeof(float) * ro * bs);
        if(!l->a) {
                fprintf(stderr, "alloc failed, exiting...");
                exit(-1);
        }
        cudaMalloc((void **)&(l->wu), sizeof(float) * ro * co);
        if(!l->wu) {
                fprintf(stderr, "alloc failed, exiting...");
                exit(-1);
        }
        cudaMalloc((void **)&(l->bu), sizeof(float) * ro * bs);
        if(!l->bu) {
                fprintf(stderr, "alloc failed, exiting...");
                exit(-1);
        }

        curandGenerator_t gen;

        curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_XORWOW);
        if(!gen) {
                fprintf(stderr, "alloc failed, exiting...");
                exit(-1);
        }
        curandSetPseudoRandomGeneratorSeed(gen, time(NULL));
        curandGenerateUniform(gen, l->w, ro * co);
        curandGenerateUniform(gen, l->b, ro * bs);
        curandDestroyGenerator(gen);
        l->rows = ro;
        l->cols = co;
        return l;
}

void layer_destroy(struct layer *l)
{
        cudaFree(l->bu);
        cudaFree(l->wu);
        cudaFree(l->a);
        cudaFree(l->wi);
        cudaFree(l->b);
        cudaFree(l->w);
        cudaFree(l);
}
