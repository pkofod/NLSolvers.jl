
function exponential(x::Vector)
    return exp((2.0 - x[1])^2) + exp((3.0 - x[2])^2)
end

function exponential_gradient!(storage::Vector, x::Vector)
    storage[1] = -2.0 * (2.0 - x[1]) * exp((2.0 - x[1])^2)
    storage[2] = -2.0 * (3.0 - x[2]) * exp((3.0 - x[2])^2)
end

function exponential_hessian!(storage::Matrix, x::Vector)
    storage[1, 1] = 2.0 * exp((2.0 - x[1])^2) * (2.0 * x[1]^2 - 8.0 * x[1] + 9)
    storage[1, 2] = 0.0
    storage[2, 1] = 0.0
    storage[2, 2] = 2.0 * exp((3.0 - x[2])^2) * (2.0 * x[2]^2 - 12.0 * x[2] + 19)
end

function nlsolvers_exponential(x, gx=nothing, hx=nothing)
	fx = exponential(x)
	if gx !== nothing
		exponential_gradient!(gx, x)
	end
	if hx !== nothing
		exponential_hessian!(hx, x)
	end
	objective_return(fx, gx, hx)
end
x0 =[0.0, 0.0]
xopt = [2.0, 3.0]
fxopt = exponential([2.0, 3.0])






##########################################################################
###
### Fletcher-Powell
###
### From [2]
### Source: A rapidly convergent descent method for minimization
###         Fletcher & Powell
##########################################################################

function fletcher_powell(x::Vector)
    function theta(x::Vector)
        if x[1] > 0
            return atan(x[2] / x[1]) / (2.0 * pi)
        else
            return (pi + atan(x[2] / x[1])) / (2.0 * pi)
        end
    end

    return 100.0 * ((x[3] - 10.0 * theta(x))^2 +
        (sqrt(x[1]^2 + x[2]^2) - 1.0)^2) + x[3]^2
end

function fletcher_powell_gradient!(storage::Vector, x::Vector)
    function theta(x::Vector)
        if x[1] > 0
            return atan(x[2] / x[1]) / (2.0 * pi)
        else
            return (pi + atan(x[2] / x[1])) / (2.0 * pi)
        end
    end

    if ( x[1]^2 + x[2]^2 == 0 )
        dtdx1 = 0;
        dtdx2 = 0;
    else
        dtdx1 = - x[2] / ( 2 * pi * ( x[1]^2 + x[2]^2 ) );
        dtdx2 =   x[1] / ( 2 * pi * ( x[1]^2 + x[2]^2 ) );
    end

    storage[1] = -2000.0*(x[3]-10.0*theta(x))*dtdx1 +
        200.0*(sqrt(x[1]^2+x[2]^2)-1)*x[1]/sqrt( x[1]^2+x[2]^2 );
    storage[2] = -2000.0*(x[3]-10.0*theta(x))*dtdx2 +
        200.0*(sqrt(x[1]^2+x[2]^2)-1)*x[2]/sqrt( x[1]^2+x[2]^2 );
    storage[3] =  200.0*(x[3]-10.0*theta(x)) + 2.0*x[3];
end

function fletcher_powell_fun_gradient!(storage::Vector, x::Vector)
    function theta(x::Vector)
        if x[1] > 0
            return atan(x[2] / x[1]) / (2.0 * pi)
        else
            return (pi + atan(x[2] / x[1])) / (2.0 * pi)
        end
    end

    if ( x[1]^2 + x[2]^2 == zero(eltype(x)) )
        dtdx1 = zero(eltype(x))
        dtdx2 = zero(eltype(x))
    else
        dtdx1 = - x[2] / ( 2.0 * pi * ( x[1]^2 + x[2]^2 ) )
        dtdx2 =   x[1] / ( 2.0 * pi * ( x[1]^2 + x[2]^2 ) )
    end

    storage[1] = -2000.0*(x[3]-10.0*theta(x))*dtdx1 +
        200.0*(sqrt(x[1]^2+x[2]^2)-1.0)*x[1]/sqrt( x[1]^2+x[2]^2 )
    storage[2] = -2000.0*(x[3]-10.0*theta(x))*dtdx2 +
        200.0*(sqrt(x[1]^2+x[2]^2)-1.0)*x[2]/sqrt( x[1]^2+x[2]^2 )
    storage[3] =  200.0*(x[3]-10.0*theta(x)) + 2.0*x[3]
    return 100.0 * ((x[3] - 10.0 * theta(x))^2 +
        (sqrt(x[1]^2 + x[2]^2) - 1.0)^2) + x[3]^2
end



function nlsolvers_fletcher_powell(x, gx=nothing, hx=nothing)
	fx = fletcher_powell(x)
	if gx !== nothing
		fletcher_powell_gradient!(gx, x)
	end
	if hx !== nothing
		fletcher_powell_hessian!(hx, x)
	end
	objective_return(fx, gx, hx)
end

x0=[-1.0, 0.0, 0.0] # Same as in source
xopt=[1.0, 0.0, 0.0]
0.0






