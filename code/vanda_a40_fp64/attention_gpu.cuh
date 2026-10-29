#pragma once
#include "attention_common.hpp"
#include <cuda_runtime.h>
#include <math_constants.h>
#include <cublas_v2.h>
#include <string>

namespace attention {
inline void cuda_check(cudaError_t e) {
    if(e!=cudaSuccess) throw std::runtime_error(std::string("CUDA: ")+cudaGetErrorString(e));
}
inline void blas_check(cublasStatus_t e) {
    if(e!=CUBLAS_STATUS_SUCCESS) throw std::runtime_error("cuBLAS status "+std::to_string(static_cast<int>(e)));
}
template<class T> struct Device {
    T* ptr=nullptr;
    explicit Device(std::size_t n) {cuda_check(cudaMalloc(reinterpret_cast<void**>(&ptr),elements(n,sizeof(T))));}
    ~Device() {if(ptr) cudaFree(ptr);}
    Device(const Device&)=delete; Device& operator=(const Device&)=delete;
};
struct Blas {
    cublasHandle_t h=nullptr;
    Blas() {
        blas_check(cublasCreate(&h));
        // Keep FP32 arithmetic; do not silently use reduced-mantissa TF32.
        blas_check(cublasSetMathMode(h,CUBLAS_PEDANTIC_MATH));
    }
    ~Blas() {if(h)cublasDestroy(h);}
};
__global__ void init_state(double* m,double* l,int n) {
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i<n) {m[i]=-CUDART_INF;l[i]=0;}
}
// One block per query row. Convert scores to exp(score-new_max), rescale
// the old unnormalized output, and update the online denominator.
__global__ void online_weights(double* s,double* out,double* m,double* l,
        int nq,int nk,int dv,int qfirst,int kfirst,bool causal) {
    int row=blockIdx.x,t=threadIdx.x;
    if(row>=nq)return;
    extern __shared__ double work[];
    __shared__ double next_max,alpha;
    double maximum=-CUDART_INF;
    for(int j=t;j<nk;j+=blockDim.x) {
        double x=s[static_cast<std::size_t>(row)*nk+j];
        if(causal && kfirst+j>qfirst+row)x=-CUDART_INF;
        s[static_cast<std::size_t>(row)*nk+j]=x;
        maximum=fmaxf(maximum,x);
    }
    work[t]=maximum;__syncthreads();
    for(int step=blockDim.x/2;step;step/=2) {
        if(t<step)work[t]=fmaxf(work[t],work[t+step]);
        __syncthreads();
    }
    if(!t) {
        next_max=fmaxf(m[row],work[0]);
        alpha=isfinite(m[row])?exp(m[row]-next_max):0.0;
        m[row]=next_max;
    }
    __syncthreads();
    double sum=0;
    for(int j=t;j<nk;j+=blockDim.x) {
        double x=s[static_cast<std::size_t>(row)*nk+j];
        double w=isfinite(x)?exp(x-next_max):0.0;
        s[static_cast<std::size_t>(row)*nk+j]=w;sum+=w;
    }
    for(int a=t;a<dv;a+=blockDim.x)out[static_cast<std::size_t>(row)*dv+a]*=alpha;
    work[t]=sum;__syncthreads();
    for(int step=blockDim.x/2;step;step/=2) {
        if(t<step)work[t]+=work[t+step];
        __syncthreads();
    }
    if(!t)l[row]=alpha*l[row]+work[0];
}
__global__ void normalize(double* out,const double* l,int n,int dv) {
    auto i=static_cast<std::size_t>(blockIdx.x)*blockDim.x+threadIdx.x;
    if(i<static_cast<std::size_t>(n)*dv)out[i]/=l[i/dv];
}
class GpuAttention {
    Options o; int n,qfirst;
    Device<double> q,out,m,l,s;
    Blas blas;
public:
    GpuAttention(const Options& opts,int local_n,int first_q,const std::vector<double>& host_q)
      :o(opts),n(local_n),qfirst(first_q),q(elements(n,o.d)),out(elements(n,o.dv)),
       m(n),l(n),s(elements(std::min(n,o.bq),std::min(o.n,o.bk))) {
        cuda_check(cudaMemcpy(q.ptr,host_q.data(),elements(host_q.size(),sizeof(double)),cudaMemcpyHostToDevice));
    }
    void reset() {
        cuda_check(cudaMemset(out.ptr,0,elements(elements(n,o.dv),sizeof(double))));
        init_state<<<(n+255)/256,256>>>(m.ptr,l.ptr,n);cuda_check(cudaGetLastError());
    }
    void consume(const double* k,const double* v,int nk_total,int kfirst) {
        for(int qi=0;qi<n;) {
            int nq=std::min(o.bq,n-qi);
            for(int ki=0;ki<nk_total;) {
                int nk=std::min(o.bk,nk_total-ki);
                const double scale=1.0/std::sqrt(static_cast<double>(o.d)),zero=0,one=1;
                // Row-major Q K^T, interpreted as its column-major transpose.
                blas_check(cublasDgemm(blas.h,CUBLAS_OP_T,CUBLAS_OP_N,nk,nq,o.d,
                    &scale,k+static_cast<std::size_t>(ki)*o.d,o.d,
                    q.ptr+static_cast<std::size_t>(qi)*o.d,o.d,&zero,s.ptr,nk));
                online_weights<<<nq,128,128*sizeof(double)>>>(s.ptr,
                    out.ptr+static_cast<std::size_t>(qi)*o.dv,m.ptr+qi,l.ptr+qi,
                    nq,nk,o.dv,qfirst+qi,kfirst+ki,o.causal);
                cuda_check(cudaGetLastError());
                blas_check(cublasDgemm(blas.h,CUBLAS_OP_N,CUBLAS_OP_N,o.dv,nq,nk,
                    &one,v+static_cast<std::size_t>(ki)*o.dv,o.dv,s.ptr,nk,
                    &one,out.ptr+static_cast<std::size_t>(qi)*o.dv,o.dv));
                ki+=nk;
            }
            qi+=nq;
        }
    }
    void finish() {
        const auto count=elements(n,o.dv);
        const auto blocks=(count+255)/256;
        if(blocks>2147483647ULL)throw std::runtime_error("Output launch grid too large");
        normalize<<<static_cast<unsigned>(blocks),256>>>(out.ptr,l.ptr,n,o.dv);
        cuda_check(cudaGetLastError());cuda_check(cudaDeviceSynchronize());
    }
    std::vector<double> download() {
        std::vector<double> host(elements(n,o.dv));
        cuda_check(cudaMemcpy(host.data(),out.ptr,elements(host.size(),sizeof(double)),cudaMemcpyDeviceToHost));
        return host;
    }
};
}
