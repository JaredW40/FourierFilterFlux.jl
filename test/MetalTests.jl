if Metal.functional()
    @testset "Metal methods" begin
        w = ConvFFT((100,), nConvDims = 1)
        @test mtl(w.fftPlan) isa Metal.MtlFFTPlan # does gpu work on the fft plans when applied directly?
        cw = gpu(w)
        @test cw.weight isa NTuple{N,MtlArray} where {N} # does gpu work on the weights?
        ### Right now the cw.fftPlan is still an FFTW plan... July 17, 2026
        @test cw.fftPlan isa Metal.MtlFFTPlan # does gpu work on the fftPlan?
        w0 = cpu(cw)
        @test w0.weight isa NTuple{N,Array} where {N} # does cpu work on the weights?
        @test w0.fftPlan isa FFTW.rFFTWPlan # does cpu work on the fftPlan?
        
        x = randn(Float32, 100)
        cx = gpu(x)
        @test cw(cx) isa MtlArray
        @test cw(cx) ≈ gpu(w(x)) rtol=1e-6 # Metal and cpu version get the same result approximately
        cw(cx)

        ∇gpu = Zygote.gradient(t -> sum(cw(t)), cx)[1]
        ∇ = Zygote.gradient(t -> sum(w(t)), x)[1]
        @test ∇ ≈ cpu(∇gpu) rtol=1e-5
        
        w1 = waveletLayer((100, 1, 1))
        cw1 = gpu(w1)
        @test cw1(cx) ≈ gpu(w1(x)) rtol=1e-4

        Metal.@allowscalar ∇gpu2 = Zygote.gradient(t -> abs(cw1(t)[1]), cx)[1]
        Metal.@allowscalar ∇2 = Zygote.gradient(t -> abs(w1(t)[1]), x)[1]
        @test (∇2) ≈ cpu(∇gpu2) rtol=1e-3
    end


    @testset "ConvFFT transform - Metal" begin
        @testset "ConvFFT 2D - CPU" begin
            originalSize = (10, 10, 1, 2)
            tmp = zeros(originalSize)
            init = zeros(Float32, originalSize)
            init[5, 5, 1, 2] = Float32(1)
            shears = ConvFFT(originalSize)
            res = shears(init)
            @test size(res) == (10, 10, 5, 1, 2)

            function minimalTransform(shears, init)
                equivalent = zeros(10, 10, 5, 1, 2)
                for i = 1:5
                    equivalent[:, :, i, :, :] = irfft(rfft(init, (1, 2)) .*
                                                    shears.weight[i], 10, (1, 2)) .+ shears.bias[i]
                end
                return equivalent
            end
            @test minimalTransform(shears, init) ≈ res

            shears = ConvFFT(originalSize, 5, abs)
            res = shears(init)
            @test abs.(minimalTransform(shears, init)) ≈ res
        end

        @testset "ConvFFT 2D - Metal" begin
            originalSize = (10, 10, 1, 2)
            init = zeros(Float32, originalSize)
            init[5, 5, 1, 2] = Float32(1)
 
            # Create CPU version first
            shears_cpu = ConvFFT(originalSize)
            res_cpu = shears_cpu(init)
 
            # Move SAME model to GPU (not creating a new one)
            init_gpu = MtlArray(init)
            shears_gpu = gpu(shears_cpu)
            res_gpu = shears_gpu(init_gpu)
 
            @test res_gpu isa MtlArray
            @test size(res_gpu) == (10, 10, 5, 1, 2)
            @test Array(res_gpu) ≈ res_cpu rtol=1e-1
        end

        @testset "ConvFFT 1D - CPU" begin
            originalSize = (10, 1, 2)
            init = zeros(Float32, originalSize)
            init[5, 1, 2] = Float32(1)
            shears = ConvFFT(originalSize, nConvDims = 1, boundary = Pad(-1))
            res = shears(init)
            @test size(res) == (10, 5, 1, 2)

            function minimalTransform(shears, init)
                equivalent = zeros(16, 5, 1, 2)
                for i = 1:5
                    equivalent[:, i, :, :] = irfft(rfft(pad(init, shears.bc.padBy), (1,)) .*
                                                shears.weight[i], 16, (1,)) .+ shears.bias[i]
                end
                return equivalent[4:13, :, :, :]
            end
            @test minimalTransform(shears, init) ≈ res

            shears = ConvFFT(originalSize, 5, abs, nConvDims = 1, boundary = Pad(-1))
            res = shears(init)
            @test abs.(minimalTransform(shears, init)) ≈ res
        end

        @testset "ConvFFT 1D - Metal" begin
            originalSize = (10, 1, 2)
            init = zeros(Float32, originalSize)
            init[5, 1, 2] = Float32(1)
 
            # Create CPU version first
            shears_cpu = ConvFFT(originalSize, nConvDims = 1, boundary = Pad(-1))
            res_cpu = shears_cpu(init)
 
            # Move SAME model to GPU
            init_gpu = MtlArray(init)
            shears_gpu = gpu(shears_cpu)
            res_gpu = shears_gpu(init_gpu)
 
            @test res_gpu isa MtlArray
            @test size(res_gpu) == (10, 5, 1, 2)
            @test cpu(res_gpu) ≈ res_cpu rtol=1e-5
        end
    end


    using FourierFilterFlux: applyWeight, applyBC, internalConvFFT
    @testset "ConvFFT constructors - Metal" begin
        @testset "Utils" begin
            explicit = [1 0 0; 0 1 0; 0 0 1; zeros(2, 3)]
            @test maximum(abs.(iden_perturbed_gaussian(5, 3) - explicit)) < 1
        end
        @testset "2D constructors - Metal" begin
            # normal size
            originalSize = (21, 11, 1, 10)
            x = randn(Float32, originalSize)
            weightMatrix = randn(Float32, (21 + 10) >> 1 + 1, 11 + 10, 1)
            weightMatrix = reshape([I zeros(16, 5)], (16, 21, 1))
            padding = (5, 5)
            shears = ConvFFT(weightMatrix, nothing, originalSize, abs,
                plan = true, boundary = Pad(padding))
            @test size(shears.fftPlan) == originalSize .+ (10, 10, 0, 0)
            @test shears.σ == abs
            @test shears.bias == nothing
            @test shears.bc.padBy == (5, 5)

            shears = ConvFFT(weightMatrix, nothing, originalSize, abs,
                plan = true, boundary = Pad(padding), trainable = true)
            trn = Flux.trainable(shears)
            @test trn[1][1] == shears.weight[1]
            @test length(trn) >= 1

            shears = ConvFFT(weightMatrix, nothing, originalSize, abs,
                boundary = Pad(padding), trainable = false)
            @test isempty(Flux.trainable(shears))

            x = randn(21, 11, 1, 10)
            ∇ = Flux.gradient((x) -> shears(x)[1, 1, 1, 1, 3], x)
            @test all(∇[1][:, :, :, [1:2..., 4:10...]] .≈ 0)

            # check that the identity ConvFFT is, in fact, an identity
            weightMatrix = ones(Float32, (21 + 10) >> 1 + 1, 11 + 10, 1)
            weightMatrix = cat(weightMatrix, weightMatrix, dims = 3)
            padding = (5, 5)
            shears = ConvFFT(weightMatrix, nothing, originalSize, identity,
                plan = true, boundary = Pad(padding))
            x = randn(Float32, 21, 11, 1, 10)
            @test shears(x)[:, :, 2, :, :] ≈ x

            # check that global multiplication in the Fourier domain is just multiplication
            weightMatrix = 2 .* ones(Float32, (21 + 10) >> 1 + 1, 11 + 10, 1)
            padding = (5, 5)
            shears = ConvFFT(weightMatrix, nothing, originalSize, identity,
                plan = true, boundary = Pad(padding))
            x = randn(21, 11, 1, 10)
            @test shears(x)[:, :, 1, :, :] ≈ 2 .* x

            # internal methods tests
            x̂ = pad(x, shears.bc.padBy)
            x̂ = shears.fftPlan * ifftshift(x̂, (1, 2))
            usedInds1 = shears.bc.padBy[1] .+ (1:size(x, 1))
            usedInds2 = shears.bc.padBy[2] .+ (1:size(x, 2))
            usedInds = (usedInds1, usedInds2)
            nextLayer = FourierFilterFlux.internalConvFFT(x̂, shears.weight, usedInds,
                shears.fftPlan, shears.bias,
                shears.analytic)
            ∇ = Flux.gradient((x̂) -> FourierFilterFlux.internalConvFFT(x̂,
                    shears.weight,
                    usedInds,
                    shears.fftPlan,
                    shears.bias, shears.analytic)[1,
                    1,
                    1,
                    1,
                    1],
                x̂)
            expected = 2.0f0 / 31 / 21
            diag_vals = abs.(diag(∇[1][:, :, 1, 1]))
            @test isapprox(diag_vals[1], expected, rtol=1e-3)
            @test all(isapprox.(diag_vals[2:end], 2 * expected, rtol=1e-3))

            ax = axes(x̂)[3:end-1]
            ∇ = Flux.gradient((x̂) -> FourierFilterFlux.applyWeight(x̂, shears.weight[1], usedInds,
                    shears.fftPlan,
                    shears.bias, FourierFilterFlux.NonAnalyticMatching())[1,
                    1,
                    1,
                    1,
                    1], x̂)
            diag_vals = abs.(diag(∇[1][:, :, 1, 1]))
            @test isapprox(diag_vals[1], expected, rtol=1e-3)
            @test all(isapprox.(diag_vals[2:end], 2 * expected, rtol=1e-3))

            ∇ = Flux.gradient((x̂) -> sum((shears.fftPlan\(x̂.*shears.weight[1]))[1:1, 1:1, 1:1, 1:1]), x̂)
            diag_vals = abs.(diag(∇[1][:, :, 1, 1]))
            @test isapprox(diag_vals[1], expected, rtol=1e-3)
            @test all(isapprox.(diag_vals[2:end], 2 * expected, rtol=1e-3))
            sheared = shears(x)
            @test size(sheared) == (21, 11, 1, 1, 10)

            weightMatrix = 2 .* ones(Float32, (21 + 10) >> 1 + 1, 11 + 10, 1)
            padding = (5, 5)
            originalSize = (21, 11, 1, 10)
            shears = ConvFFT(weightMatrix, nothing, originalSize, identity,
                plan = true, boundary = Pad(padding))


            # convert to a compatible gpu version
            gpuVer = shears |> gpu
            @test gpuVer.weight[1] isa MtlArray
            @test gpuVer.fftPlan isa AbstractFFTs.Plan
            if !(gpuVer.weight[1] isa MtlArray)
                println("gpuVer.weight is of type $(typeof(gpuVer.weight))")
            end
            if !(gpuVer.fftPlan isa AbstractFFTs.Plan)    
                println("gpuVer.fftPlan is of type $(typeof(gpuVer.fftPlan))")
            end

            # extra channel dimension
            originalSize = (20, 10, 16, 1, 10)
            shears = ConvFFT(randn(Float32, 16, 20, 3), nothing, originalSize, abs,
                plan = true, boundary = Pad((5, 5)))
            @test size(shears.fftPlan) == originalSize .+ (10, 10, 0, 0, 0)
            @test shears.σ == abs
            @test shears.bias == nothing
            @test shears.bc.padBy == (5, 5)
            wSize = originalSize[1:2] .+ (10, 10)
            wSize = (wSize[1] >> 1 + 1, wSize[2], 3)
            @test size(shears.weight[1]) == wSize[1:2]
            @test length(shears.weight) == wSize[3]

            # random initialization
            originalSize = (20, 10, 16, 1, 10)
            shears = ConvFFT(originalSize, 3, abs,
                plan = true, boundary = Pad((5, 5)))
            @test size(shears.fftPlan) == originalSize .+ (10, 10, 0, 0, 0)
            @test shears.σ == abs
            @test size(shears.bias[1]) == (originalSize[3:4]...,)
            @test length(shears.bias) == 3
            @test shears.bc.padBy == (5, 5)
            wSize = originalSize[1:2] .+ (10, 10)
            wSize = (wSize[1] >> 1 + 1, wSize[2], 3)
            @test size(shears.weight[1]) == wSize[1:2]
            @test length(shears.weight) == wSize[3]
        end


        @testset "1D constructors - Metal" begin
            # normal size
            originalSize = (21, 1, 10)
            x = randn(Float32, originalSize)
            @testset "basic tests across boundary conditions" begin
                weightMatrix = randn(Float32, (21 + 10) >> 1 + 1, 1)
                padding = (5,)
                shears = ConvFFT(weightMatrix, nothing, originalSize, abs,
                    plan = true, boundary = Pad(padding), trainable = true)
                @test size(shears.fftPlan) == originalSize .+ (10, 0, 0)
                @test shears.σ == abs
                @test shears.bias == nothing
                @test shears.bc.padBy == (5,)
                trn = Flux.trainable(shears)
                @test trn[1][1] == shears.weight[1] 
                @test length(trn) >= 1

                shears = ConvFFT(weightMatrix, nothing, originalSize, abs,
                    plan = true, boundary = Pad(padding), trainable = false)
                @test isempty(Flux.trainable(shears))

                x = randn(21, 1, 10)
                ∇ = Flux.gradient((x) -> shears(x)[1, 1, 1, 3], x)
                @test all(∇[1][:, :, [1:2..., 4:10...]] .≈ 0)

                # Sym test
                weightMatrix = randn(Float32, (21 + 1), 1)
                shears = ConvFFT(weightMatrix, nothing, originalSize, abs,
                    plan = true, boundary = FourierFilterFlux.Sym())
                @test size(shears.fftPlan) == originalSize .* (2, 1, 1)
                @test shears.σ == abs
                @test shears.bias == nothing
                @test typeof(shears.bc) <: Sym
                trn = Flux.trainable(shears)
                @test trn[1][1] == shears.weight[1]
                @test length(trn) >= 1
                x = randn(21, 1, 10)
                ∇ = Flux.gradient((x) -> shears(x)[1, 1, 1, 3], x)
                @test all(∇[1][:, :, [1:2..., 4:10...]] .≈ 0)
                @test all(abs.(∇[1][:, 1, 3]) .> 0)
                weightMatrix = randn(Float32, 21 >> 1 + 1, 1)
                shears = ConvFFT(weightMatrix, nothing, originalSize, abs,
                    plan = true, boundary = FourierFilterFlux.Periodic())
                @test size(shears.fftPlan) == originalSize
                @test shears.σ == abs
                @test shears.bias == nothing
                @test typeof(shears.bc) <: FourierFilterFlux.Periodic
                trn = Flux.trainable(shears)
                @test trn[1][1] == shears.weight[1]
                @test length(trn) >= 1
                x = randn(21, 1, 10)
                ∇ = Flux.gradient((x) -> shears(x)[1, 1, 1, 3], x)
                @test all(∇[1][:, :, [1:2..., 4:10...]] .≈ 0)
                @test all(abs.(∇[1][:, 1, 3]) .> 0)
            end

            # check that the identity ConvFFT is, in fact, an identity
            @testset "Identity tests" begin
                weightMatrix = ones(Float32, (21 + 10) >> 1 + 1, 1)
                weightMatrix = cat(weightMatrix, weightMatrix, dims = 2)
                padding = 5
                shears = ConvFFT(weightMatrix, nothing, originalSize, identity,
                    plan = true, boundary = Pad(padding))
                x = randn(Float32, 21, 1, 10)
                @test shears(x)[:, 1, :, :] ≈ x

                weightMatrix = ones(Float32, 21 + 1, 1)
                weightMatrix = cat(weightMatrix, weightMatrix, dims = 2)
                shears = ConvFFT(weightMatrix, nothing, originalSize, identity,
                    plan = true, boundary = FourierFilterFlux.Sym())
                x = randn(Float32, 21, 1, 10)
                @test shears(x)[:, 1, :, :] ≈ x
                weightMatrix = ones(Float32, 21 >> 1 + 1, 1)
                weightMatrix = cat(weightMatrix, weightMatrix, dims = 2)
                shears = ConvFFT(weightMatrix, nothing, originalSize, identity,
                    plan = true, boundary = FourierFilterFlux.Periodic())
                x = randn(Float32, 21, 1, 10)
                @test shears(x)[:, 1, :, :] ≈ x
            end

            # check that global multiplication in the Fourier domain is just multiplication
            @testset "times 2" begin

                weightMatrix = 2 .* ones(Float32, (21 + 10) >> 1 + 1, 1)
                padding = 5
                shears = ConvFFT(weightMatrix, nothing, originalSize, identity,
                    plan = true, boundary = Pad(padding))
                x = randn(21, 1, 10)
                @test shears(x)[:, 1, :, :] ≈ 2 .* x



                weightMatrix = 2 .* ones(Float32, 21 + 1, 1)
                shears = ConvFFT(weightMatrix, nothing, originalSize, identity,
                    plan = true, boundary = Sym())
                x = randn(21, 1, 10)
                @test shears(x)[:, 1, :, :] ≈ 2 .* x


                weightMatrix = 2 .* ones(Float32, 21 >> 1 + 1, 1)
                shears = ConvFFT(weightMatrix, nothing, originalSize, identity,
                    plan = true, boundary = FourierFilterFlux.Periodic())
                x = randn(Float32, 21, 1, 10)
                @test shears(x)[:, 1, :, :] ≈ 2 .* x
            end


            weight = (2 .* ones(Complex{Float32}, (21 + 10) >> 1 + 1),)
            bc = Pad(5)
            x = randn(Float32, 21, 1, 10)
            xbc, usedInds = applyBC(x, bc, 1)
            x̂ = rfft(xbc, (1,))
            fftPlan = plan_rfft(xbc, (1,))
            An = map(x -> FourierFilterFlux.NonAnalyticMatching(), (1:length(weight)...,))
            nextLayer = internalConvFFT(x̂, weight, usedInds, fftPlan, nothing, An)
            ∇ = Flux.gradient((x̂) -> internalConvFFT(x̂, weight, usedInds, fftPlan, nothing, An)[1,
                    1,
                    1,
                    1,
                    1],
                x̂)
            y, ∂ = Zygote.pullback((x̂) -> internalConvFFT(x̂, weight, usedInds, fftPlan, nothing, An)[1,
                    1,
                    1,
                    1,
                    1],
                x̂)
            ∂(y)
            ∂(y) # repeated calls to the derivative were causing errors while argWrapper
            # was in use
            # @test all(isapprox.(abs.(∇[1][:, 1, 1]), 2.0f0 / 31, rtol=1e-3))
            expected = 2.0f0 / 31
            vals = abs.(∇[1][:, 1, 1])
            @test isapprox(vals[1], expected, rtol=1e-3)
            @test all(isapprox.(vals[2:end], 2 * expected, rtol=1e-3))
            # no bias, not analytic and real valued output

            # no bias, analytic (so complex valued)
            fftPlan = plan_fft(xbc, (1,))
            ∇ = Flux.gradient((x̂) -> abs(applyWeight(x̂,
                    weight[1],
                    usedInds,
                    fftPlan,
                    nothing,
                    FourierFilterFlux.AnalyticWavelet())[1,
                    1,
                    1,
                    1]),
                x̂)
            @test all(abs.(∇[1][:, 1, 1]) .≈ 2.0f0 / 31)

            # no bias, not analytic, complex valued, but still symmetric
            real(applyWeight(x̂,
                weight[1],
                usedInds,
                fftPlan,
                nothing,
                FourierFilterFlux.RealWaveletRealSignal()))
            fftPlan = plan_fft(xbc, (1,))
            ∇ = Flux.gradient((x̂) -> real(applyWeight(x̂,
                    weight[1],
                    usedInds,
                    fftPlan,
                    nothing,
                    FourierFilterFlux.RealWaveletRealSignal())[1,
                    1,
                    1,
                    1]),
                x̂)
            @test all(abs.(∇[1][2:end, 1, 1]) .≈ 2 * 2.0f0 / 31)
            @test abs(∇[1][1, 1, 1]) ≈ 2.0f0 / 31

            # internal methods tests
            weightMatrix = 2 .* ones(Float32, (21 + 10) >> 1 + 1, 1)
            padding = 5
            shears = ConvFFT(weightMatrix, nothing, originalSize, identity,
                plan = true, boundary = Pad(padding))
            x = randn(Float32, 21, 1, 10)
            x̂ = pad(x, shears.bc.padBy)
            x̂ = shears.fftPlan * ifftshift(x̂, (1, 2))
            usedInds = (shears.bc.padBy[1] .+ (1:size(x, 1)),)
            nextLayer = FourierFilterFlux.internalConvFFT(x̂, shears.weight, usedInds,
                shears.fftPlan,
                shears.bias, shears.analytic)
            ∇ = Flux.gradient((x̂) -> FourierFilterFlux.internalConvFFT(x̂,
                    shears.weight,
                    usedInds,
                    shears.fftPlan,
                    shears.bias, shears.analytic)[1,
                    1,
                    1,
                    1,
                    1],
                x̂)
            expected = 2.0f0 / 31
            vals = abs.(∇[1][:, 1, 1])
            @test isapprox(vals[1], expected, rtol=1e-3)
            @test all(isapprox.(vals[2:end], 2 * expected, rtol=1e-3))

            # no bias, not analytic and real valued output
            # no bias, analytic (so complex valued)
            # no bias, not analytic, complex valued, but still symmetric
            # biased (and one of the others, doesn't matter which)

            ∇ = Flux.gradient((x̂) -> (shears.fftPlan\(x̂.*shears.weight[1]))[1, 1, 1, 1], x̂)
            diag_vals = abs.(∇[1][:, :, 1, 1])
            @test isapprox(diag_vals[1], expected, rtol=1e-3)
            @test all(isapprox.(diag_vals[2:end], 2 * expected, rtol=1e-3))
            sheared = shears(x)
            @test size(sheared) == (21, 1, 1, 10)

            # convert to a compatible gpu version
            gpuVer = shears |> gpu
            @test gpuVer.weight[1] isa MtlArray
            @test gpuVer.fftPlan isa AbstractFFTs.Plan

            # extra channel dimension
            originalSize = (20, 16, 1, 10)
            shears = ConvFFT(randn(Float32, 16, 3), nothing, originalSize, abs,
                plan = true, boundary = Pad(5), trainable = false)
            @test size(shears.fftPlan) == originalSize .+ (10, 0, 0, 0)
            @test shears.σ == abs
            @test shears.bias == nothing
            @test shears.bc.padBy == (5,)
            wSize = ((originalSize[1] + (10)) >> 1 + 1, 3)
            @test size(shears.weight[1]) == wSize[1:end-1]
            @test length(shears.weight) == wSize[end]

            # random initialization
            originalSize = (20, 16, 1, 10)
            shears = ConvFFT(originalSize, 3, abs,
                plan = true, boundary = Pad(5), nConvDims = 1)
            @test size(shears.fftPlan) == originalSize .+ (10, 0, 0, 0)
            @test shears.σ == abs
            @test size(shears.bias[1]) == (originalSize[2:end-1]...,)
            @test shears.bc.padBy == (5,)
            wSize = ((originalSize[1] + 10) >> 1 + 1, 3)
            @test size(shears.weight[1]) == wSize[1:end-1]
            @test length(shears.weight) == wSize[end]
        end
    end
end
