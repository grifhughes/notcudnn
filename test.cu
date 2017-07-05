#include "io.h"
#include "layer.h"
#include <cublas_v2.h>

/* io */
const int trsize = 60000;

/* net */
static const int features = 784;
static const int hidden = 1000;
static const int classes = 10;
static const int train_iter = 1000; 
static const int batch_size = 256;
static const float reg = 0.5f;
static const float lrate = 0.001f;
static const float regularization = 1.0f - ((lrate * reg) / trsize);

/* gpu */
static const int ntpb = 512; 
static const float neg = -1.0f;
static const float alph = 1.0f;
static const float bet = 0.0f;

__device__ float drelu(float x)
{
        return x > 0.0f ? 1.0f : 0.0f;
}

__global__ void gpu_relu(const float *__restrict__ A, float *__restrict__ B, int N)
{
        int idx = threadIdx.x + blockDim.x * blockIdx.x;
        if(idx < N) 
                B[idx] = fmaxf(0.0f, A[idx]);
}

__global__ void gpu_drelu(float *A, int N)
{
        int idx = threadIdx.x + blockDim.x * blockIdx.x;
        if(idx < N) 
                A[idx] = drelu(A[idx]);
}

__global__ void gpu_exp(float *A, int N)
{
        int idx = threadIdx.x + blockDim.x * blockIdx.x;
        if(idx < N) 
                A[idx] = exp(A[idx]);
}

__global__ void gpu_subscal(const float *__restrict__ A, float *__restrict__ B, float x, int N)
{
        int idx = threadIdx.x + blockDim.x * blockIdx.x;
        if(idx < N) 
                B[idx] = A[idx] - x;
}

__global__ void gpu_sm_normalize(const float *__restrict__ A, float *__restrict__ B, int N)
{
        int idx = threadIdx.x + blockDim.x * blockIdx.x;
        if(idx < N) {
                int midx = 0;
                cublasHandle_t h;
                cublasCreate(&h);
                cublasIsamax(h, classes,
                                A + idx * classes, 1,
                                &midx);
                gpu_subscal<<<(classes+ntpb-1)/ntpb,ntpb>>>(
                                A + idx * classes, 
                                B + idx * classes, 
                                *(A + (midx - 1) + idx * classes), 
                                classes
                                );
                cublasDestroy(h);
        }
}

__global__ void gpu_sm_expscal(float *A, int N)
{
        int idx = threadIdx.x + blockDim.x * blockIdx.x;
        if(idx < N) {
                float sum = 0.0f;
                cublasHandle_t h;
                cublasCreate(&h);
                cublasSasum(h, classes, 
                                A + idx * classes, 1,
                                &sum);
                sum = 1.0f / sum;
                cublasSscal(h, classes,
                                &sum,
                                A + idx * classes, 1);
                cublasDestroy(h);
        }
}

int main(void)
{
        srand(time(NULL));

        FILE *trf;
        trf = fopen("mnist_train.csv", "r");

        struct iod *io = iod_create(features, classes, trsize);
        struct layer *hid = layer_create(hidden, features, batch_size); 
        struct layer *out = layer_create(classes, hidden, batch_size); 

        cublasHandle_t handle;
        cublasCreate(&handle);

        float *dtmp;
        cudaMalloc((void **)&dtmp, sizeof(float) * hidden * batch_size); 

        iod_parse(io, trf);

        /* training loop */
        unsigned stepy = 0;
        for(int e = 0; e < train_iter; ++e) {
                /* choose random start for mini batch submatrix */
                stepy = ((trsize - batch_size) + 1) * (float)rand()/(float)RAND_MAX;

                /* forward propagation */
                cublasSgemm(handle, 
                                CUBLAS_OP_N, CUBLAS_OP_N, 
                                hidden, batch_size, features,
                                &alph, 
                                hid->w, hidden,
                                io->i + stepy * features, features,
                                &bet, 
                                hid->wi, hidden);
                cublasSgeam(handle, 
                                CUBLAS_OP_N, CUBLAS_OP_N,
                                hidden, batch_size, 
                                &alph,
                                hid->wi, hidden,
                                &alph,
                                hid->b, hidden, 
                                hid->wi, hidden);
                gpu_relu<<<(hidden*batch_size+ntpb-1)/ntpb,ntpb>>>(hid->wi,
                                hid->a,
                                hidden * batch_size);
                cublasSgemm(handle, 
                                CUBLAS_OP_N, CUBLAS_OP_N, 
                                classes, batch_size, hidden,
                                &alph, 
                                out->w, classes,
                                hid->a, hidden,
                                &bet, 
                                out->wi, classes);
                cublasSgeam(handle, 
                                CUBLAS_OP_N, CUBLAS_OP_N,
                                classes, batch_size, 
                                &alph,
                                out->wi, classes,
                                &alph,
                                out->b, classes, 
                                out->wi, classes);
                gpu_sm_normalize<<<(batch_size+ntpb-1)/ntpb,ntpb>>>(out->wi, out->a,
                                batch_size);
                gpu_exp<<<(classes*batch_size+ntpb-1)/ntpb,ntpb>>>(out->a, classes *
                                batch_size);
                gpu_sm_expscal<<<(batch_size+ntpb-1)/ntpb,ntpb>>>(out->a,
                                batch_size);

                /* backpropagation */
                cublasSgeam(handle,
                                CUBLAS_OP_N, CUBLAS_OP_N, 
                                classes, batch_size,
                                &alph,
                                out->a, classes,
                                &neg,
                                io->t + stepy * classes, classes,
                                out->bu, classes);
                cublasSgemm(handle,
                                CUBLAS_OP_N, CUBLAS_OP_T,
                                classes, hidden, batch_size,
                                &alph,
                                out->bu, classes,
                                hid->a, hidden, 
                                &bet,
                                out->wu, classes);
                cublasSgemm(handle,
                                CUBLAS_OP_T, CUBLAS_OP_N,
                                hidden, batch_size, classes,
                                &alph,
                                out->w, classes,
                                out->bu, classes,
                                &bet,
                                hid->bu, hidden);
                gpu_drelu<<<(hidden*batch_size+ntpb-1)/ntpb,ntpb>>>(hid->wi, hidden);
                cublasSgemm(handle,
                                CUBLAS_OP_N, CUBLAS_OP_N,
                                hidden, batch_size, hidden,
                                &alph,
                                hid->bu, hidden,
                                hid->wi, hidden,
                                &bet,
                                dtmp, hidden);
                cublasSgemm(handle,
                                CUBLAS_OP_N, CUBLAS_OP_T,
                                hidden, features, batch_size,
                                &alph,
                                dtmp, hidden,
                                io->i + stepy * features, features,
                                &bet,
                                hid->wu, hidden);

                /* scale gradients */
                cublasSscal(handle, classes * batch_size,
                                &lrate,
                                out->bu, 1);
                cublasSscal(handle, classes * hidden,
                                &lrate,
                                out->wu, 1);
                cublasSscal(handle, hidden * batch_size,
                                &lrate,
                                hid->bu, 1);
                cublasSscal(handle, hidden * features,
                                &lrate,
                                hid->wu, 1);

                /* regularize weights */
                cublasSscal(handle, classes * hidden,
                                &regularization,
                                out->w, 1);
                cublasSscal(handle, hidden * features,
                                &regularization,
                                hid->w, 1);

                /* update weights */
                cublasSgeam(handle,
                                CUBLAS_OP_N, CUBLAS_OP_N, 
                                classes, batch_size,
                                &alph,
                                out->b, classes,
                                &neg,
                                out->bu, classes,
                                out->b, classes);
                cublasSgeam(handle,
                                CUBLAS_OP_N, CUBLAS_OP_N, 
                                classes, hidden,
                                &alph,
                                out->w, classes,
                                &neg,
                                out->wu, classes,
                                out->w, classes);
                cublasSgeam(handle,
                                CUBLAS_OP_N, CUBLAS_OP_N, 
                                hidden, batch_size,
                                &alph,
                                hid->b, hidden,
                                &neg,
                                hid->bu, hidden,
                                hid->b, hidden);
                cublasSgeam(handle,
                                CUBLAS_OP_N, CUBLAS_OP_N, 
                                hidden, features,
                                &alph,
                                hid->w, hidden,
                                &neg,
                                hid->wu, hidden,
                                hid->w, hidden);
        }

        layer_destroy(hid);
        layer_destroy(out);
        iod_destroy(io);
        cudaFree(dtmp);
        cublasDestroy(handle);
}
