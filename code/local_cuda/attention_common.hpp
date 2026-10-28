#pragma once
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace attention {
inline std::size_t elements(std::size_t a, std::size_t b) {
    if (b && a > std::numeric_limits<std::size_t>::max()/b)
        throw std::runtime_error("Size overflow");
    return a*b;
}
inline std::uint64_t number(const std::string& s, bool zero=false) {
    if(s.empty() || s.find_first_not_of("0123456789")!=std::string::npos)
        throw std::runtime_error("Invalid unsigned integer: "+s);
    const auto n=std::stoull(s);
    if(!zero && !n) throw std::runtime_error("Expected positive integer");
    return n;
}
struct Options {
    int n=1024,d=64,dv=64,bq=128,bk=256,repeats=5;
    std::uint64_t seed=42;
    bool causal=false,validate=false,help=false;
    std::string output;
};
inline Options parse(int argc,char** argv) {
    Options o;
    for(int i=1;i<argc;++i) {
        std::string a=argv[i];
        if(a=="--causal") o.causal=true;
        else if(a=="--validate") o.validate=true;
        else if(a=="--help") o.help=true;
        else {
            if(i+1==argc) throw std::runtime_error("Missing value for "+a);
            std::string s=argv[++i];
            if(a=="--output") {o.output=s;continue;}
            auto n=number(s,a=="--seed");
            if(a=="--seed") {o.seed=n;continue;}
            if(n>static_cast<std::uint64_t>(std::numeric_limits<int>::max()))
                throw std::runtime_error("Argument exceeds INT_MAX");
            int v=static_cast<int>(n);
            if(a=="--n") o.n=v;
            else if(a=="--d") o.d=v;
            else if(a=="--dv") o.dv=v;
            else if(a=="--bq") o.bq=v;
            else if(a=="--bk") o.bk=v;
            else if(a=="--repeats") o.repeats=v;
            else throw std::runtime_error("Unknown option: "+a);
        }
    }
    return o;
}
inline double sample(std::uint64_t& state) {
    state+=UINT64_C(0x9e3779b97f4a7c15);
    auto z=state;
    z=(z^(z>>30))*UINT64_C(0xbf58476d1ce4e5b9);
    z=(z^(z>>27))*UINT64_C(0x94d049bb133111eb);
    z^=z>>31;
    return 2.0*(static_cast<double>(z>>11)/9007199254740992.0)-1.0;
}
// Jump directly into the serial generator's Q, K or V stream.
template<class T> inline std::vector<T> generate(std::uint64_t seed,
        std::uint64_t offset,std::size_t count) {
    auto state=seed+UINT64_C(0x9e3779b97f4a7c15)*offset;
    std::vector<T> a(count);
    for(auto& x:a) x=static_cast<T>(sample(state));
    return a;
}
inline int rows(int n,int p,int rank) {return n/p+(rank<n%p);}
inline int first(int n,int p,int rank) {return rank*(n/p)+std::min(rank,n%p);}
inline std::vector<double> reference(const Options& o) {
    const auto nd=elements(o.n,o.d),nv=elements(o.n,o.dv);
    // Round inputs to FP32 first: compare arithmetic, not different input values.
    auto q=generate<float>(o.seed,0,nd);
    auto k=generate<float>(o.seed,nd,nd);
    auto v=generate<float>(o.seed,2*static_cast<std::uint64_t>(nd),nv);
    std::vector<double> out(nv,0.0),s(o.n);
    const double scale=1/std::sqrt(static_cast<double>(o.d));
    for(int i=0;i<o.n;++i) {
        const int count=o.causal?i+1:o.n;
        double m=-std::numeric_limits<double>::infinity(),l=0;
        for(int j=0;j<count;++j) {
            double x=0;
            for(int a=0;a<o.d;++a)
                x+=static_cast<double>(q[static_cast<std::size_t>(i)*o.d+a])*k[static_cast<std::size_t>(j)*o.d+a];
            s[j]=x*scale;m=std::max(m,s[j]);
        }
        for(int j=0;j<count;++j) {s[j]=std::exp(s[j]-m);l+=s[j];}
        for(int j=0;j<count;++j)
            for(int a=0;a<o.dv;++a)
                out[static_cast<std::size_t>(i)*o.dv+a]+=s[j]/l*v[static_cast<std::size_t>(j)*o.dv+a];
    }
    return out;
}
inline bool validate(const std::vector<float>& got,const Options& o,
                     double& max_abs,double& rel_l2) {
    auto ref=reference(o);
    if(got.size()!=ref.size()) throw std::runtime_error("Validation size mismatch");
    long double err=0,norm=0;max_abs=0;
    for(std::size_t i=0;i<got.size();++i) {
        if(!std::isfinite(got[i])) {max_abs=rel_l2=std::numeric_limits<double>::infinity();return false;}
        double delta=static_cast<double>(got[i])-ref[i];
        max_abs=std::max(max_abs,std::abs(delta));err+=delta*delta;norm+=ref[i]*ref[i];
    }
    rel_l2=std::sqrt(static_cast<double>(err/std::max(norm,1e-30L)));
    return max_abs<=1e-4 && rel_l2<=1e-4;
}
inline void csv(const std::string& path,const std::vector<float>& a,int n,int dv) {
    std::ofstream f(path);
    if(!f) throw std::runtime_error("Cannot open output: "+path);
    f<<std::setprecision(9);
    for(int i=0;i<n;++i) {
        for(int j=0;j<dv;++j) {if(j) f<<',';f<<a[static_cast<std::size_t>(i)*dv+j];}
        f<<'\n';
    }
    f.close();if(!f) throw std::runtime_error("Output write failed");
}
}
