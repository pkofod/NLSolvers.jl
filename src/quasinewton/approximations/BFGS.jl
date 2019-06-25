# The BFGS update
# The inverse can either be updated with
#  H = (I-s*y'/y'*s)*H*(I-y*s'/y'*s)/(s'y)^2 - (H*y*s' + s*y'*H)/s'*y
# or
#  H = H + (s'*y + y'*H*y)*(s*s')/(s'*y)^2 - (B*y*s' + s*y'*H)/(s'*y)
# The direct update can be written as
#  B = B = yy'/y's-Bss'B'/s'B*s

struct BFGS{T1} <: QuasiNewton{T1}
   approx::T1
end
# function update!(scheme::BFGS, B::Inverse, Δx, y)
#    B.A = B.A + y*y'/dot(Δx, y) - B.A*y*y'*B.A/(y'*B.A*y)
# end


function update!(B, s, y, scheme::BFGS{<:Direct})
    # We could write this as
    #     B .+= (y*y')/dot(s, y) - (B*s)*(s'*B)/(s'*B*s)
    #     B .+= (y*y')/dot(s, y) - b*b'/dot(s, b)
    # where b = B*s
    # But instead, we split up the calculations. First calculate the denominator
    # in the first term
    ρ = inv(dot(s,y)) # scalar
    # Then calculate the vector b
    b = B*s # vector temporary
    # Calculate one vector divided by dot(s, b)
    ρbb = inv(dot(s, b))*b
    # And calculate
    B .+= (ρ*y)*y' .- ρbb*b'
end
function update(B, s, y, scheme::BFGS{<:Direct})
   # As above, but out of place
   ρ = inv(dot(s, y))
   b = B*s
   ρbb = inv(dot(s,b))*b
   B + (ρ*y)*y' - ρbb*b'
end
function update(H, s, y, scheme::BFGS{<:Inverse})
   sy = dot(s, y)
   ρ = inv(sy)

#   if isfinite(ρ)
      C = (I - ρ*s*y')
      H = C*H*C' + ρ*s*s'
#   end

   H
end
function update!(H, s, y, scheme::BFGS{<:Inverse})
   sy = dot(s, y)
   ρ = inv(sy)

   if isfinite(ρ)
      Hy = H*y
      H .= H .+ ((sy+y'*Hy).*ρ^2)*(s*s')
      Hys = Hy*s'
      Hys .= Hys .+ Hys'
      H .= H .- Hys.*ρ
   end
   H
end
# end
# function update!(H, s, y, scheme::BFGS{<:Inverse})
#    sy = dot(s, y)
#    ρ = inv(sy)
#
#    if isfinite(ρ)
#       C = (I - ρ.*s*y')
#       H .= C*H*C' + ρ*s*s'
#    end
#    H
# end

function update!(A::UniformScaling, s, y, scheme::BFGS{<:Inverse})
   update(A, s, y, scheme)
end
function update!(A::UniformScaling, s, y, scheme::BFGS{<:Direct})
   update(A, s, y, scheme)
end
