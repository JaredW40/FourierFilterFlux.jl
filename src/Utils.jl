import NNlib.relu
# just a little bit of type piracy used internally TODO maybe don't...
relu(x::C) where {C<:Complex} = real(x) > 0 ? x : C(0)
import Zygote: @adjoint
# ways to convert between gpu and cpu
import Adapt.adapt
export adapt

Adapt.adapt(::Type{Array{T}}, P::FFTW.FFTWPlan{T}) where {T} = P
function Adapt.adapt(::Type{Array{T}}, P::FFTW.rFFTWPlan) where {T}
    plan_rfft(zeros(real(T), P.sz), P.region)
end
Adapt.adapt(::Type{<:Array}, P::AbstractFFTs.Plan) = P

_source_length(fftPlan::AbstractFFTs.Plan) = size(fftPlan, 1)

function _unwrap_plan(p)
    p isa Tuple ? p[1] : p
end
function getBatchSize(c::ConvFFT)
    p = _unwrap_plan(c.fftPlan)
    sz = size(p)
    return sz[end]
end

"""
    weights = originalDomain()
given a ConvFFT, get the weights as represented in the time domain. optionally, apply a function σ to each pointwise afterward
"""
function originalDomain(cv::ConvFFT; σ = identity)
    w_cpu = map(w -> adapt(Array, w), cv.weight)
    p = _unwrap_plan(cv.fftPlan)
    λorig(x, an) = originalDomain(x, p, an)
    σ.(cat(map(λorig, w_cpu, cv.analytic)..., dims = 2))
end
function originalDomain(cv::ConvFFT{2}; σ = identity)
    w_cpu = map(w -> adapt(Array, w), cv.weight)
    σ.(irfft(cat(w_cpu..., dims=ndims(w_cpu[1])+1), _source_length(_unwrap_plan(cv.fftPlan)), (1, 2)))
end

originalDomain(wav, fftPlan, ::NonAnalyticMatching) = irfft(wav, _source_length(fftPlan))

function originalDomain(wav, fftPlan, ::AnalyticWavelet)
    isSourceOdd = mod(_source_length(fftPlan) + 1, 2)
    return ifft([wav; zeros(eltype(wav), size(wav, 1) - 1 - isSourceOdd)], 1)
end

function originalDomain(wav, fftPlan,
        ::Union{RealWaveletComplexSignal,RealWaveletRealSignal})
    isSourceOdd = mod(_source_length(fftPlan) + 1, 2)
    return ifft([wav; reverse(conj.(wav[2:end-isSourceOdd]))], 1)
end


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
    cpu_cat = cat(map(wi -> adapt(Array, wi), w)..., dims=3)
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
    cpu_cat = cat(map(wi -> adapt(Array, wi), w)..., dims=3)
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
    origSize = _source_length(_unwrap_plan(cv.fftPlan))

    z = dispReal ?
        apply.(irfft(cpu(cat(w..., dims=2)), origSize, (1,)))[restrict...] :
        apply.(cpu(cat(w..., dims=2)))[restrict...]
    (x, z)
end
@recipe function f(cv::ConvFFT{1}; vis = 1, dispReal = false,
        apply = abs, restrict = (Colon(), vis))
    w = cv.weight
    origSize = _source_length(_unwrap_plan(cv.fftPlan))

    z = dispReal ?
        apply.(irfft(cpu(cat(w..., dims = 2)), origSize, (1,)))[restrict...] :
        apply.(cpu(cat(w..., dims = 2)))[restrict...]

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
    sz = size(p)
    signalSize = originalSize(sz[1:ndims(l.weight[1])], l.bc)
    return (signalSize..., sz[(ndims(l.weight[1])+1):end]...)
end