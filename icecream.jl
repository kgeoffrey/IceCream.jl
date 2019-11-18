module icecream

export icecream, layers, compile, train

using LinearAlgebra
using Distributions
using StatsBase
using ReverseDiff

include("core.jl")
include("activations.jl")
include("loss.jl")
include("optimizer.jl")
include("metrics.jl")

end
