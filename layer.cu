#include "layer.h"

struct layer *layer_create(int ro, int co, int bs)
{
        struct layer *l;
        cudaMallocManaged(&l, sizeof(struct layer));
        cudaMalloc((void **)&(l->w), sizeof(float) * ro * co);
        cudaMalloc((void **)&(l->b), sizeof(float) * ro * bs);
        cudaMalloc((void **)&(l->wi), sizeof(float) * ro * bs);
        cudaMalloc((void **)&(l->a), sizeof(float) * ro * bs);
        cudaMalloc((void **)&(l->wu), sizeof(float) * ro * co);
        cudaMalloc((void **)&(l->bu), sizeof(float) * ro * bs);
        l->rows = ro;
        l->cols = co;

        curandGenerator_t gen;
        curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_XORWOW);
        curandSetPseudoRandomGeneratorSeed(gen, time(NULL));
        curandGenerateUniform(gen, l->w, ro * co);
        curandGenerateUniform(gen, l->b, ro * bs);
        curandDestroyGenerator(gen);

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
