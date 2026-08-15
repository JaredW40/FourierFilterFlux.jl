#=  The functor / trainable contract =#
const Opt = Flux.Optimisers

@testset "functor and trainable contract" begin

    @testset "ConvFFT is walkable" begin
        w = waveletLayer((128, 1, 2))
        @test !Functors.isleaf(w)

        ch = Functors.children(w)
        @test ch isa NamedTuple
        @test !isempty(ch)
        # every declared field should be reachable
        for f in fieldnames(typeof(w))
            @test haskey(ch, f)
        end
    end

    @testset "fft plans stay leaves" begin
        w = waveletLayer((128, 1, 2))
        plans = w.fftPlan isa Tuple ? w.fftPlan : (w.fftPlan,)
        for p in plans
            @test Functors.isleaf(p)
        end
    end

    @testset "trainable names a subset of the children" begin
        for trainable in (true, false)
            w = waveletLayer((128, 1, 2), trainable = trainable)
            ch = Functors.children(w)
            trn = Flux.trainable(w)
            @test trn isa NamedTuple
            for k in keys(trn)
                @test haskey(ch, k)
            end
        end
    end

    @testset "Flux.setup succeeds" begin
        wTrain = waveletLayer((128, 1, 2), trainable = true)
        st = Flux.setup(Adam(), wTrain)
        @test st isa NamedTuple
        @test haskey(st, :weight)
        # a trainable layer must actually expose parameters to the optimiser
        @test !isempty(Opt.trainables(wTrain))

        wFixed = waveletLayer((128, 1, 2), trainable = false)
        stFixed = with_logger(NullLogger()) do
            Flux.setup(Adam(), wFixed)
        end
        @test stFixed isa NamedTuple
        @test isempty(Opt.trainables(wFixed))
    end

    @testset "a gradient step actually updates the weights" begin
        w = waveletLayer((128, 1, 2), trainable = true)
        before = deepcopy(w.weight)

        x = randn(Float32, 128, 1, 2)
        opt = Flux.setup(Adam(0.1), w)
        loss(m, x) = sum(abs2, m(x))
        Flux.train!(loss, w, [(x,)], opt)

        @test any(before[i] != w.weight[i] for i in eachindex(w.weight))
        @test all(all(isfinite, wi) for wi in w.weight)

        # the plan must survive training untouched - it is not a parameter
        @test w.fftPlan isa Tuple || w.fftPlan isa AbstractFFTs.Plan
    end

    @testset "display does not error" begin
        for trainable in (true, false)
            w = waveletLayer((128, 1, 2), trainable = trainable)
            @test sprint(show, w) isa String
            @test sprint((io, x) -> show(io, MIME"text/plain"(), x), w) isa String
        end
    end
end