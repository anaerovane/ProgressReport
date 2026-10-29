#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>
#include <omp.h>
#include <mpi.h>

struct Options {
    std::size_t n = 256, d = 64, dv = 64, repeats = 3;
    int threads = 1;
    std::uint64_t seed = 42;
    bool causal = false, check = false;
    std::string output, input;
};

std::size_t product(std::size_t a, std::size_t b) {
    if (b && a > std::numeric_limits<std::size_t>::max() / b)
        throw std::runtime_error("Matrix size overflow");
    return a * b;
}

std::uint64_t integer(const std::string& value, bool allow_zero = false) {
    if (value.empty() || value.find_first_not_of("0123456789") != std::string::npos)
        throw std::runtime_error("Expected an unsigned integer: " + value);
    std::size_t used = 0;
    const auto x = std::stoull(value, &used);
    if (used != value.size() || (!allow_zero && x == 0))
        throw std::runtime_error("Invalid integer: " + value);
    return x;
}

Options parse(int argc, char** argv) {
    Options o;
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--help") {
            std::cout << "attention_openmp [--n N] [--d D] [--dv DV] [--seed S]\n"
                      << "                [--repeats R] [--threads T] [--causal] [--check]\n"
                      << "                [--input matrices.txt] [--output result.csv]\n";
            std::exit(0);
        } else if (arg == "--causal") o.causal = true;
        else if (arg == "--check") o.check = true;
        else {
            if (i + 1 >= argc) throw std::runtime_error("Missing value for " + arg);
            const std::string value = argv[++i];
            if (arg == "--input") o.input = value;
            else if (arg == "--output") o.output = value;
            else if (arg == "--seed") o.seed = integer(value, true);
            else {
                const auto x = integer(value);
                if (arg == "--threads") {
                    if (x > static_cast<std::uint64_t>(std::numeric_limits<int>::max()))
                        throw std::runtime_error("Thread count is too large");
                    o.threads = static_cast<int>(x);
                } else if (x > std::numeric_limits<std::size_t>::max()) {
                    throw std::runtime_error("Argument exceeds size_t");
                } else if (arg == "--n") o.n = static_cast<std::size_t>(x);
                else if (arg == "--d") o.d = static_cast<std::size_t>(x);
                else if (arg == "--dv") o.dv = static_cast<std::size_t>(x);
                else if (arg == "--repeats") o.repeats = static_cast<std::size_t>(x);
                else throw std::runtime_error("Unknown argument: " + arg);
            }
        }
    }
    return o;
}

double sample(std::uint64_t& state) {
    state += UINT64_C(0x9e3779b97f4a7c15);
    auto z = state;
    z = (z ^ (z >> 30)) * UINT64_C(0xbf58476d1ce4e5b9);
    z = (z ^ (z >> 27)) * UINT64_C(0x94d049bb133111eb);
    z ^= z >> 31;
    return 2.0 * (static_cast<double>(z >> 11) / 9007199254740992.0) - 1.0;
}

void attention(const std::vector<double>& q, const std::vector<double>& k,
               const std::vector<double>& v, std::vector<double>& out,
               const Options& o, int thread_count, std::size_t row_begin, std::size_t row_end) {
    const double scale = 1.0 / std::sqrt(static_cast<double>(o.d));
    bool invalid = false;
#pragma omp parallel num_threads(thread_count) reduction(||:invalid)
    {
        std::vector<double> scores(o.n);
#pragma omp for schedule(static)
        for (long long row = static_cast<long long>(row_begin); row < static_cast<long long>(row_end); ++row) {
            const std::size_t i = static_cast<std::size_t>(row);
            const std::size_t count = o.causal ? i + 1 : o.n;
            double maximum = -std::numeric_limits<double>::infinity();
            for (std::size_t j = 0; j < count; ++j) {
                double dot = 0.0;
                for (std::size_t a = 0; a < o.d; ++a)
                    dot += q[i * o.d + a] * k[j * o.d + a];
                scores[j] = dot * scale;
                if (!std::isfinite(scores[j])) invalid = true;
                maximum = std::max(maximum, scores[j]);
            }
            double denominator = 0.0;
            for (std::size_t j = 0; j < count; ++j) {
                scores[j] = std::exp(scores[j] - maximum);
                denominator += scores[j];
            }
            double* output_row = out.data() + i * o.dv;
            std::fill(output_row, output_row + o.dv, 0.0);
            for (std::size_t j = 0; j < count; ++j) {
                const double weight = scores[j] / denominator;
                const double* value_row = v.data() + j * o.dv;
                for (std::size_t a = 0; a < o.dv; ++a)
                    output_row[a] += weight * value_row[a];
            }
            for (std::size_t a = 0; a < o.dv; ++a)
                if (!std::isfinite(output_row[a])) invalid = true;
        }
    }
    if (invalid) throw std::runtime_error("Non-finite attention value");
}

void self_check() {
    Options o;
    o.n = 3; o.d = 2; o.dv = 2;
    std::vector<double> q(6, 0.0), k(6, 0.0), out(6);
    const std::vector<double> v{1, 2, 3, 4, 5, 6};
    auto close = [](double x, double expected) {
        if (std::abs(x - expected) > 1e-12 * std::max(1.0, std::abs(expected)))
            throw std::runtime_error("Self-check failed");
    };
    attention(q, k, v, out, o, 2, 0, o.n);
    for (std::size_t i = 0; i < 3; ++i) {
        close(out[2 * i], 3); close(out[2 * i + 1], 4);
    }
    o.causal = true;
    attention(q, k, v, out, o, 2, 0, o.n);
    for (std::size_t i = 0; i < 3; ++i) {
        close(out[2 * i], static_cast<double>(i + 1));
        close(out[2 * i + 1], static_cast<double>(i + 2));
    }
    std::cout << "self_check=PASS\n";
}

int mpi_initialized = 0;
int world_rank = 0;
int world_size = 1;

int main(int argc, char** argv) {
    try {
        const Options o = parse(argc, argv);
        MPI_Init(&argc, &argv); mpi_initialized = 1;
        MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
        MPI_Comm_size(MPI_COMM_WORLD, &world_size);
        if (o.check && world_rank == 0) self_check();
        omp_set_dynamic(0);
        omp_set_num_threads(o.threads);
        std::vector<double> q(product(o.n, o.d)), k(q.size());
        std::vector<double> v(product(o.n, o.dv)), out(v.size()), reference(v.size());
        const std::size_t row_begin = o.n * static_cast<std::size_t>(world_rank) / static_cast<std::size_t>(world_size);
        const std::size_t row_end = o.n * static_cast<std::size_t>(world_rank + 1) / static_cast<std::size_t>(world_size);
        if (o.input.empty()) {
            auto state = o.seed;
            for (auto* matrix : {&q, &k, &v})
                for (double& x : *matrix) x = sample(state);
        } else {
            std::ifstream in(o.input);
            if (!in) throw std::runtime_error("Cannot open input: " + o.input);
            for (auto* matrix : {&q, &k, &v})
                for (double& x : *matrix)
                    if (!(in >> x) || !std::isfinite(x))
                        throw std::runtime_error("Input needs finite Q, K, V values matching dimensions");
            std::string extra;
            if (in >> extra) throw std::runtime_error("Unexpected extra input data");
        }
        attention(q, k, v, out, o, o.threads, row_begin, row_end); // warm-up
        std::vector<double> times;
        for (std::size_t r = 0; r < o.repeats; ++r) {
            const auto start = std::chrono::steady_clock::now();
            attention(q, k, v, out, o, o.threads, row_begin, row_end);
            const auto stop = std::chrono::steady_clock::now();
            double local_ms = std::chrono::duration<double, std::milli>(stop - start).count();
            double max_ms = 0.0;
            MPI_Reduce(&local_ms, &max_ms, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
            if (world_rank == 0) times.push_back(max_ms);
        }
        Options serial_options = o;
        attention(q, k, v, reference, serial_options, 1, row_begin, row_end);
        long double error2 = 0.0L, reference2 = 0.0L, checksum = 0.0L;
        const std::size_t value_begin = row_begin * o.dv, value_end = row_end * o.dv;
        for (std::size_t i = value_begin; i < value_end; ++i) {
            const long double diff = static_cast<long double>(out[i]) - reference[i];
            error2 += diff * diff;
            reference2 += static_cast<long double>(reference[i]) * reference[i];
            checksum += out[i];
        }
        long double local_metrics[3] = {error2, reference2, checksum}, global_metrics[3] = {};
        MPI_Allreduce(local_metrics, global_metrics, 3, MPI_LONG_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        error2 = global_metrics[0]; reference2 = global_metrics[1]; checksum = global_metrics[2];
        const double relative_l2 = std::sqrt(static_cast<double>(error2 / std::max(reference2, 1e-300L)));
        if (relative_l2 > 1e-12) throw std::runtime_error("Hybrid output differs from one-thread reference");
        if (world_rank == 0) std::sort(times.begin(), times.end());
        if (world_rank == 0) {
        const double median = times.size() % 2 ? times[times.size() / 2]
            : times[times.size() / 2 - 1] / 2 + times[times.size() / 2] / 2;
        std::cout << std::setprecision(17)
                  << "implementation=mpi_openmp_fp64\nn=" << o.n << "\nd=" << o.d
                  << "\ndv=" << o.dv << "\ncausal=" << o.causal << "\nseed=" << o.seed
                  << "\nranks=" << world_size << "\nthreads_per_rank=" << o.threads << "\nrepeats=" << o.repeats
                  << "\nmedian_ms=" << median << "\nmin_ms=" << times.front()
                  << "\nmax_ms=" << times.back() << "\nrelative_l2=" << relative_l2
                  << "\nchecksum=" << checksum << '\n';
        if (!o.output.empty()) {
            if (world_size != 1) throw std::runtime_error("--output is supported only with one MPI rank");
            std::ofstream file(o.output);
            if (!file) throw std::runtime_error("Cannot open output: " + o.output);
            file << std::setprecision(17);
            for (std::size_t i = 0; i < o.n; ++i) {
                for (std::size_t a = 0; a < o.dv; ++a) {
                    if (a) file << ',';
                    file << out[i * o.dv + a];
                }
                file << '\n';
            }
        }
        }
        MPI_Finalize(); mpi_initialized = 0;
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "rank=" << world_rank << " Error: " << e.what() << '\n';
        if (mpi_initialized) MPI_Abort(MPI_COMM_WORLD, 1);
        return 1;
    }
}

