struct DFP{T1} <: QuasiNewton{T1}
   approx::T1
end
# function update!(scheme::DFP, B::Inverse, s, y)
#    B.A = B.A + s*s'/dot(s, y) - B.A*y*y'*B.A/(y'*B.A*y)
# end
# function update!(scheme::DFP, B::Direct, s, y)
#    B.A = (I - y*s'/dot(y, s))*B.A*(I - s*y'/dot(y, s)) + y*y'/dot(y, s)
# end


function update(H, s, y, scheme::DFP{<:Inverse})
    ρ = inv(dot(s, y))
    if ρ > 1e6
        H = H + ρ*s*s' - H*(y*y')*H/(y'*H*y)
    end
    H
end
function update(B, s, y, scheme::DFP{<:Direct})
    ρ = inv(dot(s, y))

    if ρ > 1e6
        C = (I - ρ*y*s')
        B = C*B*C' + ρ*y*y'
    else
        if isa(B, UniformScaling)
            B = B + zero(eltype(s))* y * s'
        end
    end
    B
end
function update!(H, s, y, scheme::DFP{<:Inverse})
    ρ = inv(dot(s, y))
    if ρ > 1e6
        H .+= ρ*s*s' - H*(y*y')*H/(y'*H*y)
    end
    H
end
function update!(B, s, y, scheme::DFP{<:Direct})
    ρ = inv(dot(s, y))
    # so right now, we just skip the update if dsy is zero
    # but we might do something else here
    if ρ > 1e6
        C = (I - ρ*y*s')
        B .= C*B*C' + ρ*y*y'
    end
    B
end
update!(A::UniformScaling, s, y , scheme::DFP{<:Inverse}) = update(A, s, y, scheme)
update!(A::UniformScaling, s, y , scheme::DFP{<:Direct}) = update(A, s, y, scheme)
