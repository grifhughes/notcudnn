#ifndef IO_H
#define IO_H

#include <string.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

static const int blk_len = 4096;
#define IDX2C(i,j,ld) (((j)*(ld))+(i))

struct iod {
        int *c;
        float *i, *t;
        int features, classes, trsize;
};

struct iod *iod_create(int f, int cl, int tr);
void iod_parse(struct iod *io, FILE *trf);
void iod_destroy(struct iod *io);

#endif
