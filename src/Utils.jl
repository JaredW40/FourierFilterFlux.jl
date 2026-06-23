import NNlib.relu
# just a little bit of type piracy used internally TODO maybe don't...
relu(x::C) where {C<:Complex} = real(x) > 0 ? x : C(0)
import CUDA: CuArray
import cuFFT
import Zygote: @adjoint
# ways to convert between gpu and cpu
import Adapt.adapt
function adapt(to, cft::ConvFFT{D,OT,F,A,V,PD,P,T,An}) where {D,OT,F,A,V,PD,P,T,An}
    cuw = map(w -> adapt(to, w), cft.weight)
    cub = adapt(to, cft.bias)
    if cft.fftPlan isa Tuple
        cuf = map(thisPlan -> adapt(to, thisPlan), cft.fftPlan)
    else
        cuf = adapt(to, cft.fftPlan)
    end
    return ConvFFT{D,OT,F,typeof(cuw),typeof(cub),PD,typeof(cuf),T,An}(cft.σ,
        cuw,
        cub,
        cft.bc,
        cuf,
        cft.analytic)
end
export adapt

import CUDA.cu
function cu(cft::ConvFFT{D,OT,F,A,V,PD,P,T,An}) where {D,OT,F,A,V,PD,P,T,An}
    origPlan = cft.fftPlan isa Tuple ? cft.fftPlan[1] : cft.fftPlan
    fullLen = _source_length(origPlan)
    cuw = Tuple(cu(_expand_weight_to_full_spectrum(w, fullLen, an)) for (w, an) in zip(cft.weight, cft.analytic))

    cub = adapt(Flux.FluxCUDAAdaptor(), cft.bias)
    if cft.fftPlan isa Tuple
        cuf = cu.(cft.fftPlan)
    else
        cuf = cu(cft.fftPlan)
    end
    return ConvFFT{D,OT,F,typeof(cuw),typeof(cub),PD,typeof(cuf),T,An}(cft.σ,
        cuw,
        cub,
        cft.bc,
        cuf,
        cft.analytic)
end

# Symmetrize a weight stored as an rfft-style half-spectrum into a full fft-style spectrum.
function _expand_weight_to_full_spectrum(w_half::AbstractArray, fullLen::Int, ::NonAnalyticMatching)
    isSourceOdd = mod(fullLen + 1, 2)
    outer = axes(w_half)[2:end]
    return cat(w_half, reverse(conj.(w_half[2:end-isSourceOdd, outer...]), dims=1), dims=1)
end
# AnalyticWavelet: CPU stores half-spectrum, zero-padded (not conjugate-reflected) to full size
function _expand_weight_to_full_spectrum(w_half::AbstractArray, fullLen::Int, ::AnalyticWavelet)
    isSourceOdd = mod(fullLen + 1, 2)
    zpad = zeros(eltype(w_half), size(w_half,1) - 1 - isSourceOdd, size(w_half)[2:end]...)
    return cat(w_half, zpad, dims=1)
end
# RealWaveletRealSignal: same conjugate-reflection as NonAnalyticMatching
function _expand_weight_to_full_spectrum(w_half::AbstractArray, fullLen::Int, ::RealWaveletRealSignal)
    isSourceOdd = mod(fullLen + 1, 2)
    outer = axes(w_half)[2:end]
    return cat(w_half, reverse(conj.(w_half[2:end-isSourceOdd, outer...]), dims=1), dims=1)
end
# RealWaveletComplexSignal: per CPU's applyWeight, the SHEAR itself (not x̂) gets conjugate-reflected
# at call time, using the same construction as RealWaveletRealSignal
function _expand_weight_to_full_spectrum(w_half::AbstractArray, fullLen::Int, ::RealWaveletComplexSignal)
    isSourceOdd = mod(fullLen + 1, 2)
    return cat(w_half, reverse(conj.(w_half[2:end-isSourceOdd]), dims=1), dims=1)
end


# Only convert FFTW plans to CUFFT plans if CUDA is actually functional
function CUDA.cu(P::FFTW.rFFTWPlan)
    if CUDA.functional()
        return cuFFT.plan_rfft(CUDA.cu(zeros(real(eltype(P)), P.sz)), P.region)
    else
        return P   # fallback to CPU FFTW plan
    end
end

function CUDA.cu(P::FFTW.cFFTWPlan)
    if CUDA.functional()
        return cuFFT.plan_fft(CUDA.cu(zeros(eltype(P), P.sz)), P.region)
    else
        return P   # fallback to CPU FFTW plan
    end
end
CUDA.cu(P::cuFFT.Plan) = P

Adapt.adapt(::Type{Array{T}}, P::FFTW.FFTWPlan{T}) where {T} = P
function Adapt.adapt(::Type{Array{T}}, P::FFTW.rFFTWPlan) where {T}
    plan_rfft(zeros(real(T), P.sz), P.region)
end
Adapt.adapt(::Type{<:Array}, P::AbstractFFTs.Plan) = P
Adapt.adapt(::Type{<:CuArray}, P::AbstractFFTs.Plan) = cu(P)
Adapt.adapt(::Flux.FluxCUDAAdaptor, P::AbstractFFTs.Plan) = cu(P)

# reredundant
adapt(::Type{<:CuArray}, x::T) where {T<:cuFFT.CuFFTPlan} = x
# is actually converting
function adapt(::Union{Type{<:Array},Flux.FluxCPUAdaptor}, 
               x::T) where {T<:cuFFT.CuFFTPlan}
    transformSize = x.output_size
    dataSize = x.input_size
    if dataSize != transformSize
        # this is an rfft, since the dimension isn't preserved
        return plan_rfft(zeros(dataSize), x.region)
    else
        return plan_fft(zeros(dataSize), x.region)
    end
end


import Flux.cpu, Flux.gpu
function gpu(x::ConvFFT{D,OT,F,A,V,PD,P,T,An}) where {D,OT,F,A,V,PD,P,T,An}
    if !CUDA.functional()
        return x
    end
    origPlan = x.fftPlan isa Tuple ? x.fftPlan[1] : x.fftPlan
    fullLen = _source_length(origPlan)

    cuw = Tuple(adapt(Flux.FluxCUDAAdaptor(), _expand_weight_to_full_spectrum(w, fullLen, an)) for (w, an) in zip(x.weight, x.analytic))
    cub = adapt(Flux.FluxCUDAAdaptor(), x.bias)
    cuf = x.fftPlan isa Tuple ? map(p -> adapt(Flux.FluxCUDAAdaptor(), p), x.fftPlan) :
                                 adapt(Flux.FluxCUDAAdaptor(), x.fftPlan)
    return ConvFFT{D,OT,F,typeof(cuw),typeof(cub),PD,typeof(cuf),T,An}(x.σ, 
        cuw, 
        cub, 
        x.bc, 
        cuf, 
        x.analytic)
end
function cpu(x::ConvFFT{D,OT,F,A,V,PD,P,T,An}) where {D,OT,F,A,V,PD,P,T,An}
    cuw = map(w -> adapt(Flux.FluxCPUAdaptor(), w), x.weight)
    cub = adapt(Flux.FluxCPUAdaptor(), x.bias)
    cuf = x.fftPlan isa Tuple ? map(p -> adapt(Flux.FluxCPUAdaptor(), p), x.fftPlan) :
                                 adapt(Flux.FluxCPUAdaptor(), x.fftPlan)
    return ConvFFT{D,OT,F,typeof(cuw),typeof(cub),PD,typeof(cuf),T,An}(x.σ, cuw, cub, x.bc, cuf, x.analytic)
end


"""
jld doesn't like the pointers required by FFTW or CuArray for fft plans, so
this creates a version which can be saved via jld. The jank format I'm using is
a tuple listing the typeo of the input (eg CuArray{Float32,3}), the input size,
and fft region.

"""
function formatJLD(cft::ConvFFT{D,OT,F,A,V,PD,P,T,An}) where {D,OT,F,A,V,PD,P,T,An}
    newPlan = formatJLD(cft.fftPlan)
    newWeight = map(w -> adapt(Flux.FluxCPUAdaptor(), w), cft.weight)
    newBias = cft.bias isa Nothing ? nothing : 
          map(b -> adapt(Flux.FluxCPUAdaptor(), b), cft.bias)
    ConvFFT{D,OT,F,typeof(newWeight),typeof(newBias),PD,typeof(newPlan),T,An}(cft.σ,
        newWeight, newBias, cft.bc,
        newPlan, cft.analytic)
end

function formatJLD(pl::Tuple)
    return ([formatJLD(x) for x in pl]...,)
end
function formatJLD(pl::AbstractFFTs.Plan)
    # ArrayType = (typeof(pl) <: CUFFT.CuFFTPlan) ? CuArray : Array
      ArrayType = (typeof(pl) <: cuFFT.CuFFTPlan) ? CuArray : Array
    return (ArrayType{eltype(pl),ndims(pl)}, size(pl), pl.region)
end
formatJLD(p) = p
"""
    weights = originalDomain()
given a ConvFFT, get the weights as represented in the time domain. optionally, apply a function σ to each pointwise afterward
"""
_source_length(fftPlan::AbstractFFTs.Plan) = size(fftPlan, 1)
_source_length(p::cuFFT.CuFFTPlan) = p.input_size[1]

function _unwrap_plan(p)
    p isa Tuple ? p[1] : p
end
function getBatchSize(c::ConvFFT)
    p = _unwrap_plan(c.fftPlan)
    sz = p isa Tuple ? p[2] :
         p isa cuFFT.CuFFTPlan ? p.input_size : p.sz
    return sz[end]
end

function originalDomain(cv::ConvFFT; σ = identity)
    w_cpu = map(w -> adapt(Flux.FluxCPUAdaptor(), w), cv.weight)
    p = _unwrap_plan(cv.fftPlan)
    λorig(x, an) = originalDomain(x, p, an)
    σ.(cat(map(λorig, w_cpu, cv.analytic)..., dims = 2))
end
function originalDomain(cv::ConvFFT{2}; σ = identity)
    w_cpu = map(w -> adapt(Flux.FluxCPUAdaptor(), w), cv.weight)
    σ.(irfft(cat(w_cpu..., dims=ndims(w_cpu[1])+1), _source_length(_unwrap_plan(cv.fftPlan)), (1, 2)))
end

originalDomain(wav, fftPlan, ::NonAnalyticMatching) = irfft(wav, _source_length(fftPlan))

function originalDomain(wav, fftPlan, ::AnalyticWavelet)
    isSourceOdd = mod(_source_length(fftPlan) + 1, 2)
    return ifft([wav; zeros(eltype(wav), size(wav, 1) - 1 - isSourceOdd)], 1)
end

function originalDomain(wav,
        fftPlan,
        ::Union{RealWaveletComplexSignal,RealWaveletRealSignal})
    isSourceOdd = mod(_source_length(fftPlan) + 1, 2)
    return ifft([wav; reverse(conj.(wav[2:end-isSourceOdd]))], 1)
end


#=
function getBatchSize(c::C) where {C<:ConvFFT}
    if typeof(c.fftPlan) <: Tuple
        return c.fftPlan[2][end]
    else
        return (typeof(c.fftPlan) <: cuFFT.CuFFTPlan ? c.fftPlan.input_size : c.fftPlan.sz)[end]
    end
end

function getBatchSize(c::ConvFFT{D,OT,A,B,C,PD,P}) where {D,OT,A,B,C,PD,P<:Tuple}
    if typeof(c.fftPlan[1]) <: Tuple
        return c.fftPlan[1][2][end]
    else
        return (typeof(c.fftPlan[1]) <: cuFFT.CuFFTPlan ? c.fftPlan[1].input_size : c.fftPlan[1].sz)[end]
    end
end
=#

function fromRestrictLocs(restrict, z, i)
    if typeof(restrict[i]) <: Colon
        return 1:size(z, i)
    else
        return restrict[i]
    end
end
@recipe function f(x, y, cv::ConvFFT{2}; vis = 1, dispReal = false,
        apply = abs, restrict = (Colon(), Colon()))
    restrict = (restrict..., vis)
    w = cv.weight
    cpu_cat = cat(map(wi -> adapt(Flux.FluxCPUAdaptor(), wi), w)..., dims=3)
    z = dispReal ?
        apply.(irfft(cpu_cat, _source_length(_unwrap_plan(cv.fftPlan)),
        (1, 2)))[restrict...] :
        apply.(ifftshift(cpu_cat, 2))[restrict...]
    (x, y, z)
end
@recipe function f(cv::ConvFFT{2}; vis = 1, dispReal = false,
        apply = abs, restrict = (Colon(), Colon()))
    restrict = (restrict..., vis)
    w = cv.weight
    cpu_cat = cat(map(wi -> adapt(Flux.FluxCPUAdaptor(), wi), w)..., dims=3)
    z = dispReal ?
        apply.(ifftshift(irfft(cpu_cat, _source_length(_unwrap_plan(cv.fftPlan)),
            (1, 2)), (1, 2)))[restrict...] :
        apply.(cpu_cat)[restrict...]
    xSz = fromRestrictLocs(restrict, z, 2)
    x = dispReal ?
        xSz :
        range(xSz.start - size(w[1], 2) / 2,
        stop = xSz.stop + size(w[1], 2) / 2,
        length = length(xSz))
    y = fromRestrictLocs(restrict, z, 1)
    (x, y, z)
end

@recipe function f(x, cv::ConvFFT{1}; vis = 1, dispReal = false,
        apply = abs, restrict = (Colon(), vis))
    w = cv.weight
    #w = map(wi -> adapt(Flux.FluxCPUAdaptor(), wi), cv.weight)
    origSize = _source_length(_unwrap_plan(cv.fftPlan))

    z = dispReal ?
        apply.(irfft(cpu(cat(w..., dims=2)), origSize, (1,)))[restrict...] :
        apply.(cpu(cat(w..., dims=2)))[restrict...]
        #apply.(irfft(cat(w..., dims=2), origSize, (1,)))[restrict...] :
        #apply.(cat(w..., dims=2))[restrict...]
    (x, z)
end
@recipe function f(cv::ConvFFT{1}; vis = 1, dispReal = false,
        apply = abs, restrict = (Colon(), vis))
    w = cv.weight
    #w = map(wi -> adapt(Flux.FluxCPUAdaptor(), wi), cv.weight)
    origSize = _source_length(_unwrap_plan(cv.fftPlan))

    z = dispReal ?
        apply.(irfft(cpu(cat(w..., dims = 2)), origSize, (1,)))[restrict...] :
        apply.(cpu(cat(w..., dims = 2)))[restrict...]
        #apply.(irfft(cat(w..., dims = 2), origSize, (1,)))[restrict...] :
        #apply.(cat(w..., dims = 2))[restrict...]

    x = 1:size(z, 1)
    (x, z)
end


"""
    positive_glorot_uniform(dims...)
same idea as a glorot_uniform, but limited to strictly positive entries.
"""
function positive_glorot_uniform(dims...)
    (rand(Float32, dims...) .* sqrt(2.0f0 / sum(Flux.nfan(dims...))))
end

"""
    uniform_perturbed_gaussian(dims...)
If there are ``n`` total entries in the matrix, each entry is gaussian distributed with a mean of ``¹/ₙ`` and a standard deviation of ``\\frac{1}{10·n}``
"""
function uniform_perturbed_gaussian(dims...)
    netSize = prod(dims)
    A = 1 ./ netSize .+ randn(dims) ./ netSize / 10
    A = Float32.(A ./ norm(A))
end
"""
    iden_perturbed_gaussian(dims...)
an identity along the diagonal with Gaussian deviations of standard deviation ``\\frac 1 {100}`` everywhere
"""
function iden_perturbed_gaussian(dims...) # only works for the 2d case
    m = minimum(dims)
    netSize = prod(dims)
    if m == dims[2]
        return [I; zeros(Float32, dims[1] - m, dims[2])] .+ randn(Float32, dims) ./ 100
    else
        return [I zeros(Float32, dims[1], dims[2] - m)] .+ randn(Float32, dims) ./ 100
    end
end
# doubly stochastic matrix (Probably more work than it's worth)


import Base: size
function size(l::ConvFFT)
    p = _unwrap_plan(l.fftPlan)
    sz = p isa Tuple ? p[2] :
         p isa cuFFT.CuFFTPlan ? p.input_size : p.sz
    signalSize = originalSize(sz[1:ndims(l.weight[1])], l.bc)
    return (signalSize..., sz[(ndims(l.weight[1])+1):end]...)
end