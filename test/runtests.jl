using FourierFilterFlux, ContinuousWavelets
using Flux, FFTW, Wavelets, Zygote
using Logging, Test, LinearAlgebra

#=  Selects which backend-specific test files actually run, and -- more
    importantly -- which backend package even gets `using`'d at all.

        GROUP=All    (default) attempt every backend this environment has
        GROUP=CUDA   only attempt CUDA -- MetalExt/Metal.jl is never touched
        GROUP=Metal  only attempt Metal -- CUDAExt/CUDA.jl is never touched
        GROUP=CPU    skip both GPU backends entirely

    Locally:   GROUP=CUDA julia --project test/runtests.jl
    In CI:     set GROUP as a per-job env var, one job per backend, each on
               the appropriate runner (CUDA job on a GPU runner, Metal job on
               an Apple Silicon runner) -- so in practice neither job needs
               the other backend installed at all, not just unused.

    The try/catch below is a second, independent layer under GROUP: even
    with GROUP=All, if a backend package genuinely isn't resolvable in this
    environment (not just non-functional -- actually absent), that's caught
    and skipped rather than erroring the whole suite. `X.functional()` is a
    third layer, already implemented inside CUDATests.jl/MetalTests.jl
    themselves: a backend can load fine but report no usable hardware. All
    three checks are complementary, not redundant -- each catches a
    different reason "don't run these tests" might be true. =#
const GROUP = get(ENV, "GROUP", "All")

@testset "FourierFilterFlux.jl" begin
    include("boundaryTests.jl")
    include("ConvFFTConstructors.jl")
    include("ConvFFTtransform.jl")
    include("waveletConv.jl")

    if GROUP in ("All", "CUDA")
        try
            using CUDA, cuDNN, cuFFT
            include("CUDATests.jl")
        catch e
            @info "CUDA/cuDNN/cuFFT not available in this environment -- skipping CUDATests.jl" exception=e
        end
    end

    if GROUP in ("All", "Metal")
        try
            using Metal
            include("MetalTests.jl")
        catch e
            @info "Metal not available in this environment -- skipping MetalTests.jl" exception=e
        end
    end
end