#ifndef LAYER_H
#define LAYER_H

#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <time.h>

struct layer {
        float *w, *b, *wi, *a,
              *wu, *bu;
        int rows, cols;
};

struct layer *layer_create(int ro, int co, int bs);
void layer_destroy(struct layer *l);

#endif
