struct ann {
        struct layer *hid, *out;
        float *dtmp;
        int batch_size;
};      

struct ann *ann_create(int f, int h, int cl, int bs)
{
        struct ann *a;
        cudaMallocManaged(&a, sizeof(struct ann));
        cudaMalloc((void **)&(a->dtmp), sizeof(float) * h * bs);
        a->hid = layer_create(h, f, bs);
        a->out = layer_create(cl, h, bs);
        a->batch_size = bs;
        return a;
}

void ann_destroy(struct ann *a)
{
        layer_destroy(a->out);
        layer_destroy(a->hid);
        cudaFree(a->dtmp);
        cudaFree(a);
}


