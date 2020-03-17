using NLSolvers, LinearAlgebra
@testset "nlsolve functionality" begin
function chebyquad(n::Integer)
    tk = 1/n

    function F!(x::Vector, fvec::Vector, fjac::Union{Nothing, Matrix})
    	if !(fvec isa Nothing)
	        fill!(fvec, 0)
	        for j = 1:n
	            temp1 = 1.0
	            temp2 = 2x[j]-1
	            temp = 2temp2
	            for i = 1:n
	                fvec[i] += temp2
	                ti = temp*temp2 - temp1
	                temp1 = temp2
	                temp2 = ti
	            end
	        end
	        iev = -1.0
	        for k = 1:n
	            fvec[k] *= tk
	            if iev > 0
	                fvec[k] += 1/(k^2-1)
	            end
	            iev = -iev
	        end
	    end
        if !isa(fjac, Nothing)
	        for j = 1:n
	            temp1 = 1.
	            temp2 = 2x[j] - 1
	            temp = 2*temp2
	            temp3 = 0.0
	            temp4 = 2.0
	            for k = 1:n
	                fjac[k,j] = tk*temp4
	                ti = 4*temp2 + temp*temp4 - temp3
	                temp3 = temp4
	                temp4 = ti
	                ti = temp*temp2 - temp1
	                temp1 = temp2
	                temp2 = ti
	            end
	        end
   		end
   		objective_return(fvec, fjac)
    end
    F!, collect(1:n)/(n+1)
end

(obj, initial) = chebyquad(6)

prob=NEqProblem(OnceDiffed(obj), nothing, NLSolvers.Euclidean(0))
res = nlsolve!(prob, copy(initial), LineSearch(Newton(), Backtracking()))
@test norm(res[2], Inf) < 1e-12
end