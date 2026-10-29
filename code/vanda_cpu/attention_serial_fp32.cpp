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

// FP64 reference: O = softmax(Q K^T / sqrt(d)) V.
// Row-major matrices; one head, batch size one, no dropout or bias.
struct Options {
    std::size_t n = 256, d = 64, dv = 64, repeats = 3;
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
            std::cout << "attention_serial [--n N] [--d D] [--dv DV] [--seed S]\n"
                      << "                 [--repeats R] [--causal] [--check]\n"
                      << "                 [--input matrices.txt] [--output result.csv]\n";
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
                if (x > std::numeric_limits<std::size_t>::max())
                    throw std::runtime_error("Argument exceeds size_t");
                if (arg == "--n") o.n = static_cast<std::size_t>(x);
                else if (arg == "--d") o.d = static_cast<std::size_t>(x);
                else if (arg == "--dv") o.dv = static_cast<std::size_t>(x);
                else if (arg == "--repeats") o.repeats = static_cast<std::size_t>(x);
                else throw std::runtime_error("Unknown argument: " + arg);
            }
        }
    }
    return o;
}

// SplitMix64 with explicitly defined conversion, reproducible across C++ runtimes.
float sample(std::uint64_t& state) {
    state += UINT64_C(0x9e3779b97f4a7c15);
    auto z = state;
    z = (z ^ (z >> 30)) * UINT64_C(0xbf58476d1ce4e5b9);
    z = (z ^ (z >> 27)) * UINT64_C(0x94d049bb133111eb);
    z ^= z >> 31;
    return 2.0 * (static_cast<float>(z >> 11) / 9007199254740992.0) - 1.0;
}

void attention(const std::vector<float>& q, const std::vector<float>& k,
               const std::vector<float>& v, std::vector<float>& out,
               std::vector<float>& scores, const Options& o) {
    const float scale = 1.0 / std::sqrt(static_cast<float>(o.d));
    std::fill(out.begin(), out.end(), 0.0);
    for (std::size_t i = 0; i < o.n; ++i) {
        const std::size_t count = o.causal ? i + 1 : o.n;
        float maximum = -std::numeric_limits<float>::infinity();
        for (std::size_t j = 0; j < count; ++j) {
            float dot = 0.0;
            for (std::size_t a = 0; a < o.d; ++a)
                dot += q[i * o.d + a] * k[j * o.d + a];
            scores[j] = dot * scale;
            if (!std::isfinite(scores[j]))
                throw std::runtime_error("Non-finite attention score; input magnitude too large");
            maximum = std::max(maximum, scores[j]);
        }
        float denominator = 0.0;
        for (std::size_t j = 0; j < count; ++j) {
            scores[j] = std::exp(scores[j] - maximum);
            denominator += scores[j];
        }
        for (std::size_t j = 0; j < count; ++j) {
            const float weight = scores[j] / denominator;
            for (std::size_t a = 0; a < o.dv; ++a)
                out[i * o.dv + a] += weight * v[j * o.dv + a];
        }
    }
    for (const float x : out)
        if (!std::isfinite(x)) throw std::runtime_error("Non-finite output");
}

// Closed-form cases exercise uniform softmax, scaling, mask and stable softmax.
void self_check() {
    Options o;
    o.n = 3; o.d = 2; o.dv = 2;
    std::vector<float> q(6, 0.0), k(6, 0.0);
    const std::vector<float> v{1, 2, 3, 4, 5, 6};
    std::vector<float> out(6), scratch(3);
    auto close = [](float x, float expected) {
        if (std::abs(x - expected) > 1e-5f * std::max(1.0f, std::abs(expected)))
            throw std::runtime_error("Self-check failed");
    };
    attention(q, k, v, out, scratch, o);
    for (std::size_t i = 0; i < 3; ++i) {
        close(out[2 * i], 3); close(out[2 * i + 1], 4);
    }
    o.causal = true;
    attention(q, k, v, out, scratch, o);
    for (std::size_t i = 0; i < 3; ++i) {
        close(out[2 * i], static_cast<float>(i + 1));
        close(out[2 * i + 1], static_cast<float>(i + 2));
    }
    o.n = 2; o.d = 2; o.dv = 1; o.causal = false;
    q = {1, 0, 1, 0}; k = {0, 0, std::sqrt(2.0), 0};
    out.resize(2); scratch.resize(2);
    attention(q, k, {0, 1}, out, scratch, o);
    close(out[0], 1.0 / (1.0 + std::exp(-1.0)));
    close(out[1], out[0]);
    // exp(1000) would overflow without subtracting the row maximum.
    k = {1000 * std::sqrt(2.0), 0, 1001 * std::sqrt(2.0), 0};
    attention(q, k, {0, 1}, out, scratch, o);
    close(out[0], 1.0 / (1.0 + std::exp(-1.0)));
    std::cout << "self_check=PASS\n";
}

int main(int argc, char** argv) {
    try {
        const Options o = parse(argc, argv);
        if (o.check) self_check();
        std::vector<float> q(product(o.n, o.d)), k(q.size());
        std::vector<float> v(product(o.n, o.dv)), out(v.size()), scores(o.n);
        if (o.input.empty()) {
            auto state = o.seed;
            for (auto* matrix : {&q, &k, &v})
                for (float& x : *matrix) x = sample(state);
        } else {
            std::ifstream in(o.input);
            if (!in) throw std::runtime_error("Cannot open input: " + o.input);
            for (auto* matrix : {&q, &k, &v})
                for (float& x : *matrix)
                    if (!(in >> x) || !std::isfinite(x))
                        throw std::runtime_error("Input needs finite Q, K, V values matching dimensions");
            std::string extra;
            if (in >> extra) throw std::runtime_error("Unexpected extra input data");
        }
        // Allocation, input generation, warm-up and output are excluded from timing.
        attention(q, k, v, out, scores, o);
        std::vector<double> times;
        for (std::size_t r = 0; r < o.repeats; ++r) {
            const auto start = std::chrono::steady_clock::now();
            attention(q, k, v, out, scores, o);
            const auto stop = std::chrono::steady_clock::now();
            times.push_back(std::chrono::duration<double, std::milli>(stop - start).count());
        }
        std::sort(times.begin(), times.end());
        const auto mid = times.size() / 2;
        const double median = times.size() % 2 ? times[mid] : times[mid - 1] / 2 + times[mid] / 2;
        long double checksum = 0.0L;
        for (float x : out) checksum += x;
        std::cout << std::setprecision(17)
                  << "implementation=serial_fp32\nn=" << o.n << "\nd=" << o.d
                  << "\ndv=" << o.dv << "\ncausal=" << o.causal << "\nseed=" << o.seed
                  << "\ninput=" << (o.input.empty() ? "generated" : o.input)
                  << "\nrepeats=" << o.repeats << "\nmedian_ms=" << median
                  << "\nmin_ms=" << times.front() << "\nmax_ms=" << times.back()
                  << "\nchecksum=" << checksum << '\n';
        if (!o.output.empty()) {
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
            file.close();
            if (!file) throw std::runtime_error("Failed writing output");
        }
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << '\n';
        return 1;
    }
}
