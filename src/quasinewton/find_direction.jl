function find_direction(A, ∇f, scheme::QuasiNewton{<:Direct})
   return -(A\∇f)
end
function find_direction(A, ∇f, scheme::QuasiNewton{<:Inverse})
   return -A*∇f
end

function find_direction!(d, B, ∇f, scheme::QuasiNewton{<:Direct})
   d .= -(B\∇f)
   d
end
function find_direction!(d, A, ∇f, scheme::QuasiNewton{<:Inverse})
   rmul!(mul!(d, A, ∇f), -1)
   d
end
