module MetalExt

using FourierFilterFlux
import FourierFilterFlux: _apply_fft, applyBC, internalConvFFT

using Metal
using Adapt
using Zygote
using AbstractFFTs, FFTW
import MLDataDevices: MetalDevice, CPUDevice

#=  Deliberately mirrors CUDAExt's design choice: never differentiate
    plan_rfft/irfft directly on MtlArray. Metal's native FFT support
    (Metal.jl 1.10+) is brand new, and its AbstractFFTs.RFFTAdjointStyle /
    IRFFTAdjointStyle coverage for real transforms is the thing under
    suspicion for the original gradient-scaling bug. Routing everything 
    through complex input + full complex fft/ifft only ever needs the simpler
    FFTAdjointStyle, which is far more likely to be complete even in a very
    new backend. If Metal's real-transform adjoints turn out to already be
    correct, this whole file could eventually be replaced by plain plan_rfft/`\` 
    calls mirroring the CPU code path -- worth revisiting once the bare 
    `rfft`/Zygote isolation test has actually been run on real Metal hardware. =#

const _mtlfft_cache_lock      = ReentrantLock()
const _mtlfft_fft_plan_cache  = Dict{Any,Any}()
const _mtlfft_ifft_plan_cache = Dict{Any,Any}()

function get_mtlfft_fft_plan(x::MtlArray, D::Int)
    key = (size(x), D, eltype(x))
    lock(_mtlfft_cache_lock) do
        get!(_mtlfft_fft_plan_cache, key) do
            plan_fft(x, 1:D)
        end
    end
end

function get_mtlfft_ifft_plan(wave::MtlArray, D::Int=1)
    key = (size(wave), eltype(wave), D)
    lock(_mtlfft_cache_lock) do
        get!(_mtlfft_ifft_plan_cache, key) do
            scaled = plan_ifft(wave, 1:D)
            (scaled.p, scaled.scale)
        end
    end
end

Zygote.@adjoint get_mtlfft_fft_plan(x, D)     = get_mtlfft_fft_plan(x, D),  _ -> (nothing, nothing)
Zygote.@adjoint get_mtlfft_ifft_plan(wave, D) = get_mtlfft_ifft_plan(wave, D), _ -> (nothing, nothing)

Zygote.@adjoint function _apply_fft(plan::Metal.MtlFFTPlan, x::MtlArray)
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
        fft_plan = get_mtlfft_fft_plan(Δc, D)
        ∂wave = conj.(scale) .* (fft_plan * Δc)
        return (nothing, nothing, ∂wave, nothing)
    end
    return y, ifft_pullback
end

FourierFilterFlux._prepare_shears(x̂::MtlArray, shears, analytic) =
    map((s, an) -> _expand_weight_to_full_spectrum(s, size(x̂, 1), an), shears, analytic)
FourierFilterFlux._source_length(x̂::MtlArray, fftPlan::Tuple{MtlArray,Int,Int}) = fftPlan[3]

Adapt.adapt(dev::MetalDevice, P::FFTW.rFFTWPlan) = Metal.functional() ? Metal.mtl(P) : P
Adapt.adapt(dev::MetalDevice, P::FFTW.cFFTWPlan) = Metal.functional() ? Metal.mtl(P) : P
Adapt.adapt(::CPUDevice, P::Metal.MtlFFTPlan) = adapt(Array, P)

# Same conjugate-reflection / zero-padding logic as CUDAExt. 
function _expand_weight_to_full_spectrum(w_half::AbstractArray, fullLen::Int, ::NonAnalyticMatching)
    isSourceOdd = mod(fullLen + 1, 2)
    outer = axes(w_half)[2:end]
    n = size(w_half, 1)
    revIdx = (n - isSourceOdd):-1:2
    cat(w_half, conj.(w_half[revIdx, outer...]), dims=1)
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
    n = size(w_half, 1)
    revIdx = (n - isSourceOdd):-1:2
    cat(w_half, conj.(w_half[revIdx, outer...]), dims=1)
end
function _expand_weight_to_full_spectrum(w_half::AbstractArray, fullLen::Int, ::RealWaveletComplexSignal)
    isSourceOdd = mod(fullLen + 1, 2)
    n = size(w_half, 1)
    revIdx = (n - isSourceOdd):-1:2
    cat(w_half, conj.(w_half[revIdx]), dims=1)
end


function (shears::ConvFFT)(x::MtlArray)
    xbc, usedInds = applyBC(x, shears.bc, ndims(shears.weight[1]))
    D = ndims(shears.weight[1])
    xbc_c = complex.(xbc)
    fwd_plan = get_mtlfft_fft_plan(xbc_c, D)
    x̂ = _apply_fft(fwd_plan, xbc_c)
    weight = FourierFilterFlux._prepare_shears(x̂, shears.weight, shears.analytic)
    nextLayer = internalConvFFT(x̂, weight, usedInds,
        (x̂, D, size(xbc, 1)), shears.bias, shears.analytic)
    return shears.σ.(nextLayer)
end
function (shears::ConvFFT{D,OT,A,B,C,PD,P})(x::MtlArray) where {D,OT,A,B,C,PD,P<:Tuple}
    xbc, usedInds = applyBC(x, shears.bc, ndims(shears.weight[1]))
    xbc_c = complex.(xbc)
    fwd_plan = get_mtlfft_fft_plan(xbc_c, D)
    x̂ = _apply_fft(fwd_plan, xbc_c)
    weight = FourierFilterFlux._prepare_shears(x̂, shears.weight, shears.analytic)
    nextLayer = internalConvFFT(x̂, weight, usedInds,
        (x̂, D, size(xbc, 1)), shears.bias, shears.analytic)
    return shears.σ.(nextLayer)
end


function FourierFilterFlux.applyWeight(x̂::MtlArray, shear, usedInds, fftPlan::Tuple{MtlArray,Int,Int}, bias::Nothing, An::NonAnalyticMatching)
    _, D, _ = fftPlan
    filtered = shear .* x̂
    inner_plan, scale = get_mtlfft_ifft_plan(filtered, D)
    tmp = real.(_apply_ifft(inner_plan, scale, filtered, D))
    @views tmp = tmp[usedInds..., axes(tmp)[length(usedInds)+1:end]...]
    return reshape(tmp, (1, size(tmp)...))
end
function FourierFilterFlux.applyWeight(x̂::MtlArray, shear, usedInds, fftPlan::Tuple{MtlArray,Int,Int}, bias::Nothing, An::AnalyticWavelet)
    _, D, _ = fftPlan
    filtered = shear .* x̂
    inner_plan, scale = get_mtlfft_ifft_plan(filtered, D)
    tmp = _apply_ifft(inner_plan, scale, filtered, D)
    @views tmp = tmp[usedInds..., axes(tmp)[length(usedInds)+1:end]...]
    return reshape(tmp, (1, size(tmp)...))
end
function FourierFilterFlux.applyWeight(x̂::MtlArray, shear, usedInds, fftPlan::Tuple{MtlArray,Int,Int}, bias::Nothing, An::RealWaveletRealSignal)
    _, D, _ = fftPlan
    tmp = shear .* x̂
    inner_plan, scale = get_mtlfft_ifft_plan(tmp, D)
    tmp = real.(_apply_ifft(inner_plan, scale, tmp, D))
    tmp = tmp[usedInds..., axes(tmp)[length(usedInds)+1:end]...]
    return reshape(tmp, (1, size(tmp)...))
end
function FourierFilterFlux.applyWeight(x̂::MtlArray, shear, usedInds, fftPlan::Tuple{MtlArray,Int,Int}, bias::Nothing, An::RealWaveletComplexSignal)
    _, D, _ = fftPlan
    tmp = shear .* x̂
    inner_plan, scale = get_mtlfft_ifft_plan(tmp, D)
    tmp = _apply_ifft(inner_plan, scale, tmp, D)
    tmp = tmp[usedInds..., axes(tmp)[length(usedInds)+1:end]...]
    return reshape(tmp, (1, size(tmp)...))
end

# Reconstruct FFTW-compatible plans from Metal plans and vice versa.
Metal.mtl(P::FFTW.rFFTWPlan) = Metal.functional() ? plan_rfft(Metal.mtl(zeros(real(eltype(P)), P.sz)), P.region) : P
Metal.mtl(P::FFTW.cFFTWPlan) = Metal.functional() ? plan_fft(Metal.mtl(zeros(eltype(P), P.sz)), P.region) : P
Metal.mtl(P::Metal.MtlFFTPlan) = P

function Adapt.adapt(::Type{<:Array}, x::Metal.MtlFFTPlan)
    sz = size(x)
    eltype(x) <: Real ? plan_rfft(zeros(real(eltype(x)), sz), 1:length(sz)) :
                         plan_fft(zeros(eltype(x), sz), 1:length(sz))
end


function Adapt.adapt_structure(dev::MetalDevice, cft::ConvFFT{D,OT,F,A,V,PD,P,T,An}) where {D,OT,F,A,V,PD,P,T,An}
    mtlw = map(w -> Metal.mtl(w), cft.weight)
    mtlb = adapt(dev, cft.bias)
    mtlf = cft.fftPlan isa Tuple ? Metal.mtl.(cft.fftPlan) : Metal.mtl(cft.fftPlan)
    return ConvFFT{D,OT,F,typeof(mtlw),typeof(mtlb),PD,typeof(mtlf),T,An}(
        cft.σ, mtlw, mtlb, cft.bc, mtlf, cft.analytic)
end

# A<:Tuple{Vararg{<:MtlArray}} is what lets this coexist with CUDAExt's own
# ::CPUDevice method for the same generic Adapt.adapt_structure function. 
function Adapt.adapt_structure(::CPUDevice,
        cft::ConvFFT{D,OT,F,A,V,PD,P,T,An}) where {D,OT,F,A<:Tuple{Vararg{<:MtlArray}},V,PD,P,T,An}
    cpw = map(w -> adapt(Array, w), cft.weight)
    cpb = adapt(Array, cft.bias)
    cpf = cft.fftPlan isa Tuple ? map(p -> adapt(Array, p), cft.fftPlan) : adapt(Array, cft.fftPlan)
    return ConvFFT{D,OT,F,typeof(cpw),typeof(cpb),PD,typeof(cpf),T,An}(
        cft.σ, cpw, cpb, cft.bc, cpf, cft.analytic)
end

end # module