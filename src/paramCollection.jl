Flux.@layer ConvFFT

function Flux.trainable(CFT::ConvFFT{D,OT,F,A,V,PD,P,true,An}) where {D,OT,F,A,V,PD,P,An}
    (; weight = CFT.weight, bias = CFT.bias)
end
function Flux.trainable(::ConvFFT{D,OT,F,A,V,PD,P,false,An}) where {D,OT,F,A,V,PD,P,An}
    NamedTuple()
end
