@testset "ConvFFT transform" begin
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
    
    if CUDA.functional()
        @testset "ConvFFT 2D - GPU" begin
            originalSize = (10, 10, 1, 2)
            init = zeros(Float32, originalSize)
            init[5, 5, 1, 2] = Float32(1)
            
            # Create CPU version first
            shears_cpu = ConvFFT(originalSize)
            res_cpu = shears_cpu(init)
            
            # Move SAME model to GPU (not creating a new one)
            init_gpu = CuArray(init)
            shears_gpu = gpu(shears_cpu)
            res_gpu = shears_gpu(init_gpu)
            
            @test res_gpu isa CuArray
            @test size(res_gpu) == (10, 10, 5, 1, 2)
            
            # Now they should match because it's the same weights
            @test cpu(res_gpu) ≈ res_cpu rtol=1e-5
        end
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
    
    if CUDA.functional()
        @testset "ConvFFT 1D - GPU" begin
            originalSize = (10, 1, 2)
            init = zeros(Float32, originalSize)
            init[5, 1, 2] = Float32(1)
            
            # Create CPU version first
            shears_cpu = ConvFFT(originalSize, nConvDims = 1, boundary = Pad(-1))
            res_cpu = shears_cpu(init)
            
            # Move SAME model to GPU
            init_gpu = CuArray(init)
            shears_gpu = gpu(shears_cpu)
            res_gpu = shears_gpu(init_gpu)
            
            @test res_gpu isa CuArray
            @test size(res_gpu) == (10, 5, 1, 2)
            
            # Now they should match
            @test cpu(res_gpu) ≈ res_cpu rtol=1e-5
        end
    end
end
