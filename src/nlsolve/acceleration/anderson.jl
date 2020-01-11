#===============================================================================
  Anderson acceleration is a fixed point iteration acceleration method. A fixed
  point problem is one of finding a solution to G(x) = x or alternatively to
  find a solution to F(x) = G(x) - x = 0. To allow for easy switching between
  methods, we write the code in the F(x) = 0 form, but it really is of no impor-
  tance. This makes it a bit simpler to keep a consistent naming scheme as well,
  since convergence is measured in F(x) not G(x) directly.
===============================================================================#
struct Anderson{T}
  delay::T
  memory::T
end
Anderson() = Anderson(1)
function _vv_shift!(G)
  for i = 1:length(G)-1
    G[i] = G[i+1]
  end
end

function nlsolve!(F::NonDiffed, x, anderson::Anderson; Gx = similar(x),
                                                       Fx = similar(x)
                                                       f_abstol=sqrt(eltype(x)))

  #==============================================================
    Do initial function iterations; default is a delay of 0,
    but we always do at least one evaluation of G, so set up AA.
  ==============================================================#
  for i = 1:anderson.delay+1
    Fx = F(x, Fx)
    if norm(Fx) < f_abstol
      return x, Fx, 0
    end
  end

  Q =
  R =
  Gx .= Fx .+ x
  x_cache = similar(x)

  #== Start Anderson Acceleration ==#
  effective_memory = 0

  n = length(x)
  m = min(n, anderson.memory)
  G = [similar(x) for i=1:m)]
  Gold = similar(x)
  for k = 1:itermax
    # # check that all values are finite
    # check_isfinite(fx)
    Δf .= .-Fx
    Gold .= Gx

    Fx = F(x, Fx)
    Gx .= Fx .+ x

    #==     Fx_k - Fx_{k-1}     ==#
    Δf .+= Fx
    Δg .+= Gx .- Gold

    #===========================================================================
     if effective memory counter is below the maximum
     memory length, use the iteration index. Otherwise, update from the right.
    ===========================================================================#
    m_eff = m_eff + 1
    # define current Q and R matrices
    Gv, Qv, Rv = view(G, :, 1:m_eff), view(Q, :, 1:m_eff), UpperTriangular(view(R, 1:m_eff, 1:m_eff))

    if m_eff <= anderson.memory
      #== we've not yet exhausted the memory, so overwrite G[m_eff] ==#
      G[m_eff] .= Δg
    else
      #=========================================================================
       We've exhausted the memory, so we need to shift all vectors one place.
      =========================================================================#
      _vv_shift!(Gv)
      #== ... and overwrite the last element ==#
      G[end] .= Δg
    end

    #==
     We need to delete a column from the right in Q and the last
     row/column in R first, if the effective memory counter is larger than the
     maximum memory length.
     ==#
    if m_eff > m
        m_eff = m_eff - 1
        Gv, Qv, Rv = view(G, :, 1:m_eff), view(Q, :, 1:m_eff), UpperTriangular(view(R, 1:m_eff, 1:m_eff))
        qrdelete!(Qv, Rv)
    end
    # And then we add a column TO R
    for i = 1:m_eff-1
        R[i, m_eff] = dot(Q[i], Δf)
        Δf .= Δf .- R[i, m_eff].*Q[i]
    end
    R[m_eff, m_eff] = norm(Δf, 2)
    # And a column to Q
    Q[:, m_eff] .= Δf./R[m_eff, m_eff]

    if !isa(droptol, Nothing)
        condR = cond(Rv)
        while condR > droptol && m_eff > 1
            qrdelete!(Qv[m_eff], Rv[m_eff])
            _vv_shift!(Gv)
            m_eff = m_eff - 1
            # define current Q and R matrices
            Gv, Qv, Rv = view(G, :, 1:m_eff), view(Q, :, 1:m_eff), UpperTriangular(view(R, 1:m_eff, 1:m_eff))
            condR = cond(Rv)
        end
    end



    #    γ[1:m_eff] .= Rv[m_eff]\Qv[m_eff]'*fcur
    γv = [m_eff]
    ldiv!(Rv, mul!(γv, Qv', Fx))

    x .= Gx .- mul!(x_cache, Gv, γv)

    if !isa(beta, Nothing)
        x .= x .- (1 .- beta).*(Fx .- Qv*Rv*γv))) # this is suboptimal!
    end

    #==  Shift  ==#
    Gx .= Gz
    x .= z
    Gz = G(z, Gz)
    if norm(Fx, Inf) < f_abstol
       return x, true, k
    end

end


function anderson!(g, x::AbstractArray{T}, cache; itermax = 1000, delay=0, tol=sqrt(eps(T)), droptol=T(1e10), beta=nothing) where T
    @unpack G, Gv, R, Rv, Rcv, Q, Qv, fold, fcur, Δf, gold, gcur, Δg, m, γ, m_eff_cache, x_cache = cache

    # Since we always do one function iteration we can store the current values
    # in gold and fold. We've already checked for convergence.
    gold .= gcur
    fold .= fcur
    # m_eff is the current history length (<= cache)
    m_eff = 0
    verbose = false
    # start acceleration
    for k = 1:itermax
        #gcur .= g(x)
        g(gcur, x)
        fcur .= gcur .- x
        verbose && println()
        verbose && println("$k) Residual: ", norm(fcur, Inf))
        if norm(fcur, Inf) < tol
            return x, true, k
        end
        Δf .= fcur .- fold
        Δg .= gcur .- gold

        # now that we have the difference, we can update the cache arrays
        # for f and g
        copyto!(fold, fcur)
        copyto!(gold, gcur)

        # if effective memory counter is below the maximum
        # memory length, use the iteration index. Otherwise, update from the
        # right.
        m_eff = m_eff + 1
        # define current Q and R matrices
        Qv, Rv = view(Q, :, 1:m_eff), UpperTriangular(view(R, 1:m_eff, 1:m_eff))

        if m_eff <= m
        G[:, m_eff] = Δg
        else
            for i = 1:size(G, 1)
                for j = 2:m
                    G[i, j-1] = G[i, j]
                end
            end
            G[:, end] = Δg
        end
        # increment effective memory counter

        # if m_eff == 1 simply update R[1, 1] and Q[:, 1], else we need to
        # loop over rows in R[:, m_eff] and columns in Q. Potentially, we
        # need to delete a column from the right in Q and the last row/column
        # in R first, if the effective memory counter is larger than the
        # maximum memory length.
        if m_eff > 1
            if m_eff > m
                m_eff = m_eff - 1
                # define current Q and R matrices
                Qv, Rv = view(Q, :, 1:m_eff), UpperTriangular(view(R, 1:m_eff, 1:m_eff))

                qrdelete!(Qv, Rv)
            end
            for i = 1:m_eff-1
                R[i, m_eff] = dot(Q[i], Δf)
#                Δf .= Δf .- R[i, m_eff].*Q[:, i]
                for j = 1:length(Δf)
                    Δf[j] = Δf[j] - R[i, m_eff]*Q[j, i]
                end
            end
        end
       R[m_eff, m_eff] = norm(Δf, 2)
       Q[:, m_eff] .= Δf./R[m_eff, m_eff]


        if !isa(droptol, Nothing)
            condR = cond(Rv)
            while condR > droptol && m_eff > 1
                qrdelete!(Qv, Rv)
                for j = 2:m_eff
                    G[j-1] = G[j]
                end
                m_eff = m_eff - 1
                # define current Q and R matrices
                Qv, Rv = view(Q, :, 1:m_eff), UpperTriangular(view(R, 1:m_eff, 1:m_eff))

                condR = cond(Rv)
            end
        end

        #    γ[1:m_eff] .= Rv[m_eff]\Qv[m_eff]'*fcur
        ldiv!(γ[m_eff], Rv, mul!(m_eff_cache[m_eff], Qv[m_eff]', fcur))
        x .= gcur .- mul!(x_cache, Gv[m_eff], γ[m_eff])
        if !isa(beta, Nothing)
            x .= x .- (1 .- beta).*(fcur .- mul!(x_cache, Qv[m_eff], mul!(m_eff_cache[m_eff], Rv[m_eff], γ[m_eff])))
        end
    end
    printstyled("!!!!!!!!!\n"; color=9)
    printstyled("!Failure!\n"; color=9)
    printstyled("!!!!!!!!!\n"; color=9)
    printstyled("Exited with\n"; color=9)
    printstyled(" * residual inf-norm $(norm(fcur, Inf))\n"; color=9)
    printstyled(" * effective memory length of $m_eff\n"; color=9)
    printstyled("and\n"; color=9)
    printstyled(" - ran for $delay initial function iterations\n"; color=9)
    printstyled(" - ran for $itermax accelerated iterations\n"; color=9)

    x, false, itermax
end
