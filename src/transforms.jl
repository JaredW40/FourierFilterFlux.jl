# TODO: version that doesn't have an fft built in
import Zygote
using Adapt

# ifft: scale .* (inner_plan * wave), adjoint w.r.t wave is conj(scale) .* fft(Δ)
_apply_fft(plan, x) = plan * x

Zygote.@adjoint function _apply_fft(plan, x)
    y = _apply_fft(plan, x)
    function fft_pullback(Δ)
        N = size(x, 1)
        inv_plan = inv(plan)
        ∂x = (inv_plan * Δ) .* N
        return (nothing, ∂x)
    end
    return y, fft_pullback
end
_apply_ifft(inner_plan, scale, wave) = scale .* (inner_plan * wave)
Zygote.@adjoint function _apply_ifft(inner_plan, scale, wave)
    y = _apply_ifft(inner_plan, scale, wave)
    function ifft_pullback(Δ)
        Δc = eltype(Δ) <: Real ? complex.(Δ) : Δ
        fft_plan = plan_fft(Δc, 1)
        ∂wave = conj.(scale) .* (fft_plan * Δc)
        return (nothing, nothing, ∂wave)
    end
    return y, ifft_pullback
end

_prepare_shears(x̂::Array, shears) = shears

# CPU single-plan fft method: 
function (shears::ConvFFT)(x::Array)
    if !(shears.weight[1] isa Array) && (x isa Array)
        error("GPU weights but CPU input — move both to the same device")
    end
    xbc, usedInds = applyBC(x, shears.bc, ndims(shears.weight[1]))
    F = shears.fftPlan
    if F isa Nothing
        F = makePlan(eltype(x), outType(shears), shears.w, size(x), shears.bc)
    end
    Forward = F isa Tuple ? F[1] : F
    if size(xbc) != size(Forward)
        xbc = reshape(xbc, size(Forward))
    end
    x̂ = Forward * xbc
    weight = _prepare_shears(x̂, shears.weight)
    nextLayer = internalConvFFT(x̂, weight, usedInds, F, shears.bias, shears.analytic)
    return shears.σ.(nextLayer)
end

# These are for the case that there are more than one fft plan:
# CPU tuple-plan method: 
function (shears::ConvFFT{D,OT,A,B,C,PD,P})(x::Array) where {D,OT,A,B,C,PD,P<:Tuple}
    if !(shears.weight[1] isa Array) && (x isa Array)
        error("GPU weights but CPU input — move both to the same device")
    end
    xbc, usedInds = applyBC(x, shears.bc, ndims(shears.weight[1]))
    Forward = shears.fftPlan[1]
    if size(xbc) != size(Forward)
        xbc = reshape(xbc, size(Forward))
    end
    x̂ = Forward * xbc
    weight = _prepare_shears(x̂, shears.weight) 
    nextLayer = internalConvFFT(x̂, weight, usedInds,
        shears.fftPlan[2], shears.bias, shears.analytic)
    return shears.σ.(nextLayer)
end


abstract type TransformTypes end # If the input/wavelets are either real or
# analytic, there are efficiency gains to be had. 

struct AnalyticWavelet <: TransformTypes end
struct RealWaveletRealSignal <: TransformTypes end
struct RealWaveletComplexSignal <: TransformTypes end
struct NonAnalyticMatching <: TransformTypes end

function internalConvFFT(x̂, shears, usedInds, fftPlan, bias, isAnalytic)
    N = ndims(shears[1])
    function łλ(ii, bias)
        @views shearAccess = shears[ii]
        @views applyWeight(x̂, shearAccess, usedInds, fftPlan, bias[ii], isAnalytic[ii])
    end
    @views function łλ(ii, bias::Nothing)
        applyWeight(x̂, shears[ii], usedInds, fftPlan, bias, isAnalytic[ii])
    end
    @views mapped = map(ii -> łλ(ii, bias), 1:length(shears))
    return permutedims(cat(mapped..., dims = 1), ((2:N+1)..., 1, (N+2):ndims(mapped[1])...))
end

# Helper to avoid the divergent CPU/GPU logic: 
function _source_length(x̂::Array, fftPlan)
    size(fftPlan, 1)  # original signal length stored in plan. 
end



# NonAnalyticMatching: no bias, not analytic and both match (either both real or both complex)
function applyWeight(x̂::Array, shear, usedInds, fftPlan, bias::Nothing, An::NonAnalyticMatching)
    tmp = fftPlan \ (shear .* x̂) # filter
    @views tmp = tmp[usedInds..., axes(tmp)[length(usedInds)+1:end]...] # get rid of the padding
    tmp = reshape(tmp, (1, size(tmp)...)) # add a dummy dimension to join over
    return tmp
end

# AnalyticWavelet: no bias, analytic wavelet (so complex valued, but only the positive half of x̂ matters)
function applyWeight(x̂::Array, shear, usedInds, fftPlan, bias::Nothing, An::AnalyticWavelet)
    outer = axes(x̂)[ndims(shear)+1:end]
    isSourceOdd = mod(_source_length(x̂, fftPlan) + 1, 2)
    accessedAxes = axes(shear)
    @views tmp = shear .* x̂[accessedAxes..., outer...] # filter
    wave = cat(tmp, adapt(tmp, zeros(eltype(tmp), size(shear, 1)-1-isSourceOdd, size(tmp)[2:end]...)), dims=1) # symmetrize
    tmp = fftPlan \ wave # back to time domain
    @views tmp = tmp[usedInds..., axes(tmp)[length(usedInds)+1:end]...] # get rid of the padding
    tmp = reshape(tmp, (1, size(tmp)...)) # add a dummy dimension to join over
    return tmp
end

# RealWaveletRealSignal: no bias, not analytic, still symmetric (i.e. real, this is for the averaging wavelet where the other wavelets are analytic/complex)
function applyWeight(x̂::Array, shear, usedInds, fftPlan, bias::Nothing, An::RealWaveletRealSignal)
    outer = axes(x̂)[ndims(shear)+1:end]
    isSourceOdd = mod(_source_length(x̂, fftPlan) + 1, 2)
    tmp = shear .* x̂ # filter
    wave = cat(tmp, reverse(conj.(tmp[2:end-isSourceOdd, outer...]), dims = 1), dims = 1) # symmetrize
    tmp = fftPlan \ wave # back to time domain
    tmp = tmp[usedInds..., axes(tmp)[length(usedInds)+1:end]...] # get rid of the padding
    tmp = reshape(tmp, (1, size(tmp)...)) # add a dummy dimension to join over
    return tmp
end

# RealWaveletComplexSignal: no bias, signal asymmetric/complex, but the wavelet is real (averaging function)
function applyWeight(x̂::Array, shear, usedInds, fftPlan, bias::Nothing, An::RealWaveletComplexSignal)
    isSourceOdd = mod(_source_length(x̂, fftPlan) + 1, 2)
    tmp = cat(shear, reverse(conj.(shear[2:end-isSourceOdd]), dims = 1), dims = 1) # construct full wavelet
    tmp = tmp .* x̂ # filter
    tmp = fftPlan \ tmp # back to time domain
    tmp = tmp[usedInds..., axes(tmp)[length(usedInds)+1:end]...] # get rid of the padding
    tmp = reshape(tmp, (1, size(tmp)...)) # add a dummy dimension to join over
    return tmp
end

# biased (and one of the others, doesn't matter which)
function applyWeight(x̂, shear, usedInds, fftPlan, bias, An)
    return applyWeight(x̂, shear, usedInds, fftPlan, nothing, An) .+ bias
end