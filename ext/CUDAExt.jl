module CUDAExt
 
using FourierFilterFlux
import FourierFilterFlux: _apply_fft, applyBC, internalConvFFT
 
using CUDA, cuFFT
using Adapt
using Zygote
using AbstractFFTs, FFTW
import MLDataDevices: CUDADevice, CPUDevice

#=  Tell AbstractFFTs how to differentiate through a bare cuFFT plan. Purely 
    defensive, as the actual hot path below never differentiates
    `Forward * x`/`fftPlan \ y` directly, it routes everything through the
    hand-rolled _apply_fft/_apply_ifft adjoints a few lines down, specifically
    to avoid depending on cuFFT's rfft/irfft adjoint styles being complete. =#
AbstractFFTs.AdjointStyle(::cuFFT.CuFFTPlan) = AbstractFFTs.FFTAdjointStyle()


const _cufft_cache_lock      = ReentrantLock()
const _cufft_fft_plan_cache  = Dict{Any, Any}()
const _cufft_ifft_plan_cache = Dict{Any, Any}()

function get_cufft_fft_plan(x::CuArray, D::Int)
    key = (size(x), D, eltype(x))
    lock(_cufft_cache_lock) do
        get!(_cufft_fft_plan_cache, key) do
            cuFFT.plan_fft(x, 1:D)
        end
    end
end
function get_cufft_ifft_plan(wave::CuArray, D::Int=1)
    key = (size(wave), eltype(wave), D)
    lock(_cufft_cache_lock) do
        get!(_cufft_ifft_plan_cache, key) do
            scaled = cuFFT.plan_ifft(wave, 1:D)
            (scaled.p, scaled.scale)
        end
    end
end

Zygote.@adjoint get_cufft_fft_plan(x, D)  = get_cufft_fft_plan(x, D),  _ -> (nothing, nothing)
Zygote.@adjoint get_cufft_ifft_plan(wave, D) = get_cufft_ifft_plan(wave, D), _ -> (nothing, nothing)


Zygote.@adjoint function _apply_fft(plan::cuFFT.CuFFTPlan, x::CuArray)
    y = _apply_fft(plan, x)
    function fft_pullback(Δ)
        N = size(x, 1)
        ∂x = (inv(plan) * Δ) .* N
        return (nothing, ∂x)
    end
    return y, fft_pullback
end
_apply_ifft(inner_plan, scale, wave, D) = scale .* (inner_plan * wave)
Zygote.@adjoint function _apply_ifft(inner_plan, scale, wave, D)
    y = _apply_ifft(inner_plan, scale, wave, D)
    function ifft_pullback(Δ)
        Δc = eltype(Δ) <: Real ? complex.(Δ) : Δ
        fft_plan = get_cufft_fft_plan(Δc, D)
        ∂wave = conj.(scale) .* (fft_plan * Δc)
        return (nothing, nothing, ∂wave, nothing)
    end
    return y, ifft_pullback
end

FourierFilterFlux._prepare_shears(x̂::CuArray, shears, analytic) = map((s, an) -> _expand_weight_to_full_spectrum(s, size(x̂, 1), an), shears, analytic)
FourierFilterFlux._source_length(x̂::CuArray, fftPlan::Tuple{CuArray,Int,Int}) = fftPlan[3]

Adapt.adapt(dev::CUDADevice, P::FFTW.rFFTWPlan) = CUDA.functional() ? CUDA.cu(P) : P
Adapt.adapt(dev::CUDADevice, P::FFTW.cFFTWPlan) = CUDA.functional() ? CUDA.cu(P) : P
Adapt.adapt(::CPUDevice, P::cuFFT.CuFFTPlan)    = adapt(Array, P)


#=  Expand a half-spectrum (rfft-style) weight into a full-length spectrum so it can 
    be multiplied elementwise against a full complex FFT of the (complex-cast) input. =#
function _expand_weight_to_full_spectrum(w_half::AbstractArray, fullLen::Int, ::NonAnalyticMatching)
    isSourceOdd = mod(fullLen + 1, 2)
    outer = axes(w_half)[2:end]
    cat(w_half, reverse(conj.(w_half[2:end-isSourceOdd, outer...]), dims=1), dims=1)
end
function _expand_weight_to_full_spectrum(w_half::AbstractArray, fullLen::Int, ::AnalyticWavelet)
    isSourceOdd = mod(fullLen + 1, 2)
    zpad = Zygote.ignore_derivatives() do
        zp = similar(w_half, size(w_half, 1) - 1 - isSourceOdd, size(w_half)[2:end]...)
        fill!(zp, zero(eltype(w_half)))
        zp
    end
    cat(w_half, zpad, dims=1)
end
function _expand_weight_to_full_spectrum(w_half::AbstractArray, fullLen::Int, ::RealWaveletRealSignal)
    isSourceOdd = mod(fullLen + 1, 2)
    outer = axes(w_half)[2:end]
    cat(w_half, reverse(conj.(w_half[2:end-isSourceOdd, outer...]), dims=1), dims=1)
end
function _expand_weight_to_full_spectrum(w_half::AbstractArray, fullLen::Int, ::RealWaveletComplexSignal)
    isSourceOdd = mod(fullLen + 1, 2)
    cat(w_half, reverse(conj.(w_half[2:end-isSourceOdd]), dims=1), dims=1)
end


#=  Forward pass on GPU: cast to complex, full FFT, elementwise filter, full IFFT. =#
# GPU single-plan fft method: 
function (shears::ConvFFT)(x::CuArray)
    xbc, usedInds = applyBC(x, shears.bc, ndims(shears.weight[1]))
    D = ndims(shears.weight[1])
    xbc_c = complex.(xbc)
    fwd_plan = get_cufft_fft_plan(xbc_c, D)
    x̂ = _apply_fft(fwd_plan, xbc_c)
    weight = FourierFilterFlux._prepare_shears(x̂, shears.weight, shears.analytic)
    nextLayer = internalConvFFT(x̂, weight, usedInds, (x̂, D, size(xbc, 1)), shears.bias, shears.analytic)
    return shears.σ.(nextLayer)
end
# GPU tuple-plan method: 
function (shears::ConvFFT{D,OT,A,B,C,PD,P})(x::CuArray) where {D,OT,A,B,C,PD,P<:Tuple}
    xbc, usedInds = applyBC(x, shears.bc, ndims(shears.weight[1]))
    xbc_c = complex.(xbc)
    fwd_plan = get_cufft_fft_plan(xbc_c, D)
    x̂ = _apply_fft(fwd_plan, xbc_c)
    weight = FourierFilterFlux._prepare_shears(x̂, shears.weight, shears.analytic)
    nextLayer = internalConvFFT(x̂, weight, usedInds, (x̂, D, size(xbc, 1)), shears.bias, shears.analytic)
    return shears.σ.(nextLayer)
end


# Per-transform-type filtering on GPU, all operating on the pre-expanded full spectrum (see _expand_weight_to_full_spectrum above)
# NonAnalyticMatching: no bias, not analytic and both match (either both real or both complex)
function FourierFilterFlux.applyWeight(x̂::CuArray, shear, usedInds, fftPlan::Tuple{CuArray,Int,Int}, bias::Nothing, An::NonAnalyticMatching)
    _, D, _ = fftPlan
    filtered = shear .* x̂
    inner_plan, scale = get_cufft_ifft_plan(filtered, D)
    tmp = real.(_apply_ifft(inner_plan, scale, filtered, D))
    @views tmp = tmp[usedInds..., axes(tmp)[length(usedInds)+1:end]...]
    return reshape(tmp, (1, size(tmp)...))
end
# AnalyticWavelet: no bias, analytic wavelet (so complex valued, but only the positive half of x̂ matters)
function FourierFilterFlux.applyWeight(x̂::CuArray, shear, usedInds, fftPlan::Tuple{CuArray,Int,Int}, bias::Nothing, An::AnalyticWavelet)
    _, D, _ = fftPlan
    filtered = shear .* x̂
    inner_plan, scale = get_cufft_ifft_plan(filtered, D)
    tmp = _apply_ifft(inner_plan, scale, filtered, D)
    @views tmp = tmp[usedInds..., axes(tmp)[length(usedInds)+1:end]...]
    return reshape(tmp, (1, size(tmp)...))
end
# RealWaveletRealSignal: no bias, not analytic, still symmetric (i.e. real, this is for the averaging wavelet where the other wavelets are analytic/complex)
function FourierFilterFlux.applyWeight(x̂::CuArray, shear, usedInds, fftPlan::Tuple{CuArray,Int,Int}, bias::Nothing, An::RealWaveletRealSignal)
    _, D, _ = fftPlan
    tmp = shear .* x̂
    inner_plan, scale = get_cufft_ifft_plan(tmp, D)
    tmp = real.(_apply_ifft(inner_plan, scale, tmp, D))
    tmp = tmp[usedInds..., axes(tmp)[length(usedInds)+1:end]...]
    return reshape(tmp, (1, size(tmp)...))
end
# RealWaveletComplexSignal: no bias, signal asymmetric/complex, but the wavelet is real (averaging function)
function FourierFilterFlux.applyWeight(x̂::CuArray, shear, usedInds, fftPlan::Tuple{CuArray,Int,Int}, bias::Nothing, An::RealWaveletComplexSignal)
    _, D, _ = fftPlan
    tmp = shear .* x̂
    inner_plan, scale = get_cufft_ifft_plan(tmp, D)
    tmp = _apply_ifft(inner_plan, scale, tmp, D)
    tmp = tmp[usedInds..., axes(tmp)[length(usedInds)+1:end]...]
    return reshape(tmp, (1, size(tmp)...))
end


# Reconstruct FFTW-compatible plans from cuFFT plans and vice versa. 
CUDA.cu(P::FFTW.rFFTWPlan) = CUDA.functional() ? cuFFT.plan_rfft(CUDA.cu(zeros(real(eltype(P)), P.sz)), P.region) : P
CUDA.cu(P::FFTW.cFFTWPlan) = CUDA.functional() ? cuFFT.plan_fft(CUDA.cu(zeros(eltype(P), P.sz)), P.region) : P
CUDA.cu(P::cuFFT.Plan) = P
 
function Adapt.adapt(::Type{<:Array}, x::T) where {T<:cuFFT.CuFFTPlan}
    dataSize = x.input_size
    x.input_size != x.output_size ? plan_rfft(zeros(real(eltype(x)), dataSize), x.region) :
                                    plan_fft(zeros(eltype(x), dataSize), x.region)
end

#=  Device transfer, dispatched on the *device type*, not on ConvFFT's own unconstrained 
    type parameters. CUDADevice/CPUDevice are concrete types owned by MLDataDevices, so 
    this cannot collide with MetalExt's methods for the same generic Adapt.adapt_structure 
    function - except in the CPUDevice direction, where both extensions target the same
    `::CPUDevice` argument. The `A<:Tuple{Vararg{CuArray}}` constraint here (and the 
    mirrored `A<:Tuple{Vararg{MtlArray}}` constraint MetalExt must use) is what 
    disambiguates those two methods from each other.=#
function Adapt.adapt_structure(dev::CUDADevice, cft::ConvFFT{D,OT,F,A,V,PD,P,T,An}) where {D,OT,F,A,V,PD,P,T,An}
    cuw = map(w -> CUDA.cu(w), cft.weight)
    cub = adapt(dev, cft.bias)
    cuf = cft.fftPlan isa Tuple ? CUDA.cu.(cft.fftPlan) : CUDA.cu(cft.fftPlan)
    return ConvFFT{D,OT,F,typeof(cuw),typeof(cub),PD,typeof(cuf),T,An}(cft.σ, cuw, cub, cft.bc, cuf, cft.analytic)
end
 
function Adapt.adapt_structure(::CPUDevice, cft::ConvFFT{D,OT,F,A,V,PD,P,T,An}) where {D,OT,F,A<:Tuple{Vararg{CuArray}},V,PD,P,T,An}
    cpw = map(w -> adapt(Array, w), cft.weight)
    cpb = adapt(Array, cft.bias)
    cpf = cft.fftPlan isa Tuple ? map(p -> adapt(Array, p), cft.fftPlan) : adapt(Array, cft.fftPlan)
    return ConvFFT{D,OT,F,typeof(cpw),typeof(cpb),PD,typeof(cpf),T,An}(cft.σ, cpw, cpb, cft.bc, cpf, cft.analytic)
end

end # module