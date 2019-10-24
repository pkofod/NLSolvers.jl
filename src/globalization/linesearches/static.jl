struct Static{T}
  α::T
end

find_steplength(ls::Static, φ, λ::T) where T = T(ls.α), φ(T(ls.α)), true
