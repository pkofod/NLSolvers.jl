struct SimulatedAnnealing{Tn, Ttemp} <: RandomSearch
    neighbor::Tn
    temperature::Ttemp
end
Base.summary(::SimulatedAnnealing) = "Simulated Annealing"

"""
# SimulatedAnnealing
## Constructor
```julia
SimulatedAnnealing(; neighbor = default_neighbor, temperature = log_temperature)
```

The constructor takes two keywords:
* `neighbor = a(x_now, [x_candidate])`, a function that generates a new iterate based on the current
* `temperature = b(iteration)`, a function of the current iteration that returns a temperature

## Description
Simulated Annealing is a derivative free method for optimization. It is based on the
Metropolis-Hastings algorithm that was originally used to generate samples from a
thermodynamics system, and is often used to generate draws from a posterior when doing
Bayesian inference. As such, it is a probabilistic method for finding the minimum of a
function, often over a quite large domains. For the historical reasons given above, the
algorithm uses terms such as cooling, temperature, and acceptance probabilities.
"""
SimulatedAnnealing(;neighbor = default_neighbor,
                    temperature = log_temperature) =
  SimulatedAnnealing(neighbor, temperature)

log_temperature(t) = 1 / log(t)^2

function default_neighbor(x_best)
  T = eltype(x_best)
  n = length(x_best)
  return x_best .+ T.(RandomNumbers.randn(n))
end

function minimize(objective, x0, method::SimulatedAnnealing; maxiter=1000)
    T = eltype(x0)


    x_best = copy(x0)
    f_best = objective(nothing, x_best)
    x_now = copy(x0)
    f_now = f_best
    for iter = 1:maxiter
        # Determine the temperature for current iteration
        t = method.temperature(iter)

        # Randomly generate a neighbor of our current state
        x_candidate = method.neighbor(x_best)

        # Evaluate the cost function at the proposed state
        f_candidate = objective(nothing, x_candidate)

        if f_candidate <= f_now
            # If proposal is superior, we always move to it
            x_now = copy(x_candidate)
            f_now = f_candidate

            # If the new state is the best state yet, keep a record of it
            if f_candidate < f_best
                x_best = copy(x_now)
                f_best = f_now
            end
        else
            # If proposal is inferior, we move to it with probability p
            p = exp(-(f_candidate - f_now) / t)
            if T(RandomNumbers.rand()) <= p
                x_now = copy(x_candidate)
                f_now = f_candidate
            end
        end
    end
    (f_best=f_best, x_best=x_best, f_now=f_now, x_now=x_now, temperature=method.temperature(maxiter))
end
