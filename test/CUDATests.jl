if CUDA.functional()
    @testset "CUDA methods" begin
        w = ConvFFT((100,), nConvDims = 1)
        @test cu(w.fftPlan) isa cuFFT.CuFFTPlan
        cw = cu(w)
        @test cw.weight isa NTuple{N,CuArray} where {N}
        @test cw.fftPlan isa cuFFT.CuFFTPlan
        cw1 = gpu(w)
        @test cw1.weight isa NTuple{N,CuArray} where {N}
        @test cw1.fftPlan isa cuFFT.CuFFTPlan 
        w1 = cpu(cw)
        @test w1.weight isa NTuple{N,Array} where {N}
        @test w1.fftPlan isa FFTW.rFFTWPlan

        x = randn(Float32, 100)
        cx = cu(x)
        @test cw(cx) isa CuArray
        @test cw(cx) ≈ cu(w(x)) rtol=1e-6
        cw(cx)

        ∇cu = Zygote.gradient(t -> sum(cw(t)), cx)[1] 
        ∇ = Zygote.gradient(t -> sum(w(t)), x)[1]
        @test ∇ ≈ cpu(∇cu) rtol=1e-5
        w1 = waveletLayer((100, 1, 1))
        cw1 = cu(w1)
        @test cw1(cx) ≈ cu(w1(x)) rtol=1e-4

        ∇_gpu = Zygote.gradient(t -> sum(abs.(cw1(t)[1:1])), cx)[1]
        ∇_cpu = Zygote.gradient(t -> sum(abs.(w1(t)[1:1])), x)[1]
        @test (∇_cpu /2)≈ cpu(∇_gpu) rtol=1e-3
    end
end