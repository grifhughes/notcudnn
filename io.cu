#include "io.h"

struct iod *iod_create(int f, int cl, int tr)
{
        struct iod *io;
        cudaMallocManaged(&io, sizeof(struct iod));
        cudaMallocManaged(&(io->i), sizeof(float) * f * tr);
        cudaMallocManaged(&(io->t), sizeof(float) * cl * tr);
        cudaMallocManaged(&(io->c), sizeof(int) * tr);
        cudaMemset((void **)&(io->t), 0, sizeof(float) * cl * tr);
        io->features = f;
        io->classes = cl;
        io->trsize = tr;
        return io;
}

void iod_parse(struct iod *io, FILE *trf)
{
        char *l;
        char line[blk_len];
        float sum, mean, sdev_sum, stddev;
        for(int z = 0; z < io->trsize; ++z) {
                fgets(line, blk_len, trf);
                l = strtok(line, ",");
                io->t[IDX2C(z, atoi(l), io->trsize)] = 1.0f;
                l = strtok(NULL, ",");

                sum = 0.0f;
                for(int j = 0; j < io->features; ++j) {
                        io->i[IDX2C(z, j, io->trsize)] = atof(l);
                        sum += io->i[IDX2C(z, j, io->trsize)];
                        l = strtok(NULL, ",");
                }
                mean = sum / io->features;
                sdev_sum = 0.0f;
                for(int k = 0; k < io->features; ++k) {
                        io->i[IDX2C(z, k, io->trsize)] -= mean;
                        sdev_sum += io->i[IDX2C(z, k, io->trsize)] * io->i[IDX2C(z, k,
                                        io->trsize)];
                }
                stddev = sqrt(sdev_sum / io->features);
                for(int g = 0; g < io->features; ++g) 
                        io->i[IDX2C(z, g, io->trsize)] /= stddev;
        }
}

void iod_destroy(struct iod *io)
{
        cudaFree(io->c);
        cudaFree(io->t); 
        cudaFree(io->i); 
        cudaFree(io); 
}
