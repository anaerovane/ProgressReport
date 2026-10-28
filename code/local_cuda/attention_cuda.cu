#include "attention_gpu.cuh"
#include <chrono>
#include <cstdio>
#include <iostream>
#include <memory>
#ifdef ATTENTION_MPI
#include <mpi.h>
static void mpi_check(int status) {
    if(status!=MPI_SUCCESS) {
        char text[MPI_MAX_ERROR_STRING];int n;
        MPI_Error_string(status,text,&n);
        throw std::runtime_error(std::string("MPI: ")+std::string(text,n));
    }
}
#endif
using namespace attention;
struct HostBuffer {
    float* ptr=nullptr;
    explicit HostBuffer(std::size_t n) {
        cuda_check(cudaMallocHost(reinterpret_cast<void**>(&ptr),elements(n,sizeof(float))));
    }
    ~HostBuffer() {if(ptr)cudaFreeHost(ptr);}
    HostBuffer(const HostBuffer&)=delete;HostBuffer& operator=(const HostBuffer&)=delete;
};
static double seconds() {
    return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
}
struct Timing {double total=0,upload=0,compute=0,mpi=0;};
int main(int argc,char** argv) {
    int rank=0,p=1,local_rank=0;
#ifdef ATTENTION_MPI
    int provided;
    if(MPI_Init_thread(&argc,&argv,MPI_THREAD_FUNNELED,&provided)!=MPI_SUCCESS)return 1;
    if(provided<MPI_THREAD_FUNNELED) {MPI_Abort(MPI_COMM_WORLD,1);return 1;}
    MPI_Comm_set_errhandler(MPI_COMM_WORLD,MPI_ERRORS_RETURN);
    MPI_Comm_rank(MPI_COMM_WORLD,&rank);MPI_Comm_size(MPI_COMM_WORLD,&p);
#endif
    try {
        const auto o=parse(argc,argv);
        if(o.help) {
            if(!rank)std::cout<<"Options: --n N --d D --dv DV --bq BQ --bk BK --seed S\n"
                <<"         --repeats R --causal --validate --output result.csv\n"
                <<"FP32, one head, batch=1, generated input only. MPI uses host staging.\n";
#ifdef ATTENTION_MPI
            MPI_Finalize();
#endif
            return 0;
        }
        if(p>o.n)throw std::runtime_error("MPI ranks must not exceed sequence length");
        if(o.n>std::numeric_limits<int>::max()-255)
            throw std::runtime_error("N too large for launch indexing");
        if(o.validate && o.n>4096)
            throw std::runtime_error("--validate limited to N<=4096: FP64 reference has quadratic CPU cost");
        const int n=rows(o.n,p,rank),qfirst=first(o.n,p,rank),maxrows=rows(o.n,p,0);
        const auto nd=elements(o.n,o.d),kv_capacity=elements(maxrows,static_cast<std::size_t>(o.d)+o.dv);
#ifdef ATTENTION_MPI
        if(kv_capacity>static_cast<std::size_t>(std::numeric_limits<int>::max()) ||
           elements(o.n,o.dv)>static_cast<std::size_t>(std::numeric_limits<int>::max()))
            throw std::runtime_error("MPI count/displacement exceeds INT_MAX");
        MPI_Comm local;
        mpi_check(MPI_Comm_split_type(MPI_COMM_WORLD,MPI_COMM_TYPE_SHARED,rank,MPI_INFO_NULL,&local));
        mpi_check(MPI_Comm_rank(local,&local_rank));
#endif
        int devices=0;cuda_check(cudaGetDeviceCount(&devices));
        if(!devices)throw std::runtime_error("No visible CUDA GPU");
        const int device=local_rank%devices;
        cuda_check(cudaSetDevice(device));
        cudaDeviceProp prop;cuda_check(cudaGetDeviceProperties(&prop,device));
        char bus[32]={};cuda_check(cudaDeviceGetPCIBusId(bus,sizeof(bus),device));
#ifdef ATTENTION_MPI
        int lp;mpi_check(MPI_Comm_size(local,&lp));
        std::vector<char> buses(static_cast<std::size_t>(lp)*sizeof(bus));
        mpi_check(MPI_Allgather(bus,sizeof(bus),MPI_CHAR,buses.data(),sizeof(bus),MPI_CHAR,local));
        for(int a=0;a<lp;++a)for(int b=a+1;b<lp;++b)
            if(std::string(buses.data()+a*sizeof(bus))==std::string(buses.data()+b*sizeof(bus)))
                throw std::runtime_error("Two local MPI ranks selected the same GPU; fix device visibility/binding");
        mpi_check(MPI_Comm_free(&local));
#endif
        // Use C stdio for per-rank diagnostics: remote MPI ranks crashed in
        // libstdc++ while constructing an ostream sentry on the Vanda nodes.
        std::printf("rank=%d gpu=%s pci_bus=%s local_rows=%d first_row=%d\n",
                    rank,prop.name,bus,n,qfirst);
        std::fflush(stdout);
        int status=0;
        // Destroy CUDA allocations/handles before finalizing MPI.
        {
            auto q=generate<float>(o.seed,elements(qfirst,o.d),elements(n,o.d));
            auto k=generate<float>(o.seed,nd+elements(qfirst,o.d),elements(n,o.d));
            auto v=generate<float>(o.seed,2*static_cast<std::uint64_t>(nd)+elements(qfirst,o.dv),elements(n,o.dv));
            HostBuffer a(kv_capacity),b(kv_capacity);
            Device<float> kv(kv_capacity);
            GpuAttention engine(o,n,qfirst,q);
            auto iteration=[&]() {
                // Input construction and synchronization barrier excluded from timing.
                std::copy(k.begin(),k.end(),a.ptr);
                std::copy(v.begin(),v.end(),a.ptr+k.size());
                cuda_check(cudaDeviceSynchronize());
#ifdef ATTENTION_MPI
                mpi_check(MPI_Barrier(MPI_COMM_WORLD));
#endif
                Timing time;const double start=seconds();
                engine.reset();cuda_check(cudaDeviceSynchronize());
                float* current=a.ptr;float* next=b.ptr;
                for(int step=0;step<p;++step) {
                    const int owner=(rank-step+p)%p,kr=rows(o.n,p,owner);
                    const auto kc=elements(kr,o.d),vc=elements(kr,o.dv);
                    double t=seconds();
                    cuda_check(cudaMemcpy(kv.ptr,current,elements(kc+vc,sizeof(float)),cudaMemcpyHostToDevice));
                    time.upload+=seconds()-t;
                    t=seconds();
                    engine.consume(kv.ptr,kv.ptr+kc,kr,first(o.n,p,owner));
                    cuda_check(cudaDeviceSynchronize());
                    time.compute+=seconds()-t;
#ifdef ATTENTION_MPI
                    if(step+1<p) {
                        const int next_owner=(owner-1+p)%p;
                        const auto incoming=elements(rows(o.n,p,next_owner),static_cast<std::size_t>(o.d)+o.dv);
                        t=seconds();
                        mpi_check(MPI_Sendrecv(current,static_cast<int>(kc+vc),MPI_FLOAT,(rank+1)%p,17,
                            next,static_cast<int>(incoming),MPI_FLOAT,(rank-1+p)%p,17,
                            MPI_COMM_WORLD,MPI_STATUS_IGNORE));
                        time.mpi+=seconds()-t;std::swap(current,next);
                    }
#endif
                }
                engine.finish();time.total=seconds()-start;
                return time;
            };
            iteration(); // warm-up, including the MPI ring
            std::vector<double> totals,uploads,computes,comms;
            for(int rep=0;rep<o.repeats;++rep) {
                auto t=iteration();
                double local[4]={t.total,t.upload,t.compute,t.mpi},global[4];
#ifdef ATTENTION_MPI
                mpi_check(MPI_Reduce(local,global,4,MPI_DOUBLE,MPI_MAX,0,MPI_COMM_WORLD));
#else
                std::copy(local,local+4,global);
#endif
                if(!rank) {
                    totals.push_back(global[0]*1000);uploads.push_back(global[1]*1000);
                    computes.push_back(global[2]*1000);comms.push_back(global[3]*1000);
                    std::cout<<std::setprecision(10)<<"sample="<<rep<<" total_ms="<<global[0]*1000
                        <<" max_upload_ms="<<global[1]*1000<<" max_compute_ms="<<global[2]*1000
                        <<" max_mpi_ms="<<global[3]*1000<<'\n';
                }
            }
            const auto local_out=engine.download();
            for(float x:local_out)if(!std::isfinite(x))throw std::runtime_error("Nonfinite GPU output");
            std::vector<float> full;
            if(o.validate || !o.output.empty()) {
#ifdef ATTENTION_MPI
                std::vector<int> counts(p),offsets(p);
                for(int r=0;r<p;++r) {
                    counts[r]=static_cast<int>(elements(rows(o.n,p,r),o.dv));
                    offsets[r]=static_cast<int>(elements(first(o.n,p,r),o.dv));
                }
                if(!rank)full.resize(elements(o.n,o.dv));
                mpi_check(MPI_Gatherv(local_out.data(),counts[rank],MPI_FLOAT,
                    rank==0?full.data():nullptr,counts.data(),offsets.data(),MPI_FLOAT,0,MPI_COMM_WORLD));
#else
                full=local_out;
#endif
            }
            double checksum=0;for(float x:local_out)checksum+=x;
#ifdef ATTENTION_MPI
            double sum;mpi_check(MPI_Reduce(&checksum,&sum,1,MPI_DOUBLE,MPI_SUM,0,MPI_COMM_WORLD));
            checksum=sum;
#endif
            if(!rank) {
                auto median=[](std::vector<double> x) {
                    std::sort(x.begin(),x.end());auto m=x.size()/2;
                    return x.size()%2?x[m]:x[m-1]/2+x[m]/2;
                };
                std::cout<<std::setprecision(10)<<"implementation="<<(p==1?"cuda_fp32":"mpi_ring_cuda_fp32")
                    <<"\nn="<<o.n<<"\nd="<<o.d<<"\ndv="<<o.dv<<"\nranks="<<p
                    <<"\nbq="<<o.bq<<"\nbk="<<o.bk<<"\nseed="<<o.seed<<"\ncausal="<<o.causal
                    <<"\nrepeats="<<o.repeats<<"\nmedian_ms="<<median(totals)
                    <<"\nmin_ms="<<*std::min_element(totals.begin(),totals.end())
                    <<"\nmax_ms="<<*std::max_element(totals.begin(),totals.end())
                    <<"\nchecksum="<<checksum<<'\n';
                if(o.validate) {
                    double abs,rel;bool pass=validate(full,o,abs,rel);
                    std::cout<<"max_abs_error="<<abs<<"\nrelative_l2_error="<<rel
                        <<"\nvalidation="<<(pass?"PASS":"FAIL")<<'\n';
                    if(!pass)status=2;
                }
                if(!o.output.empty())csv(o.output,full,o.n,o.dv);
            }
#ifdef ATTENTION_MPI
            mpi_check(MPI_Bcast(&status,1,MPI_INT,0,MPI_COMM_WORLD));
#endif
        }
#ifdef ATTENTION_MPI
        MPI_Finalize();
#endif
        return status;
    } catch(const std::exception& e) {
        std::cerr<<"rank="<<rank<<" error="<<e.what()<<'\n';
#ifdef ATTENTION_MPI
        MPI_Abort(MPI_COMM_WORLD,1);
#endif
        return 1;
    }
}
