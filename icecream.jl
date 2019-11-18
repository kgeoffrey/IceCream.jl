module Icecream

export icecream, layers, compile, train!, dense,
    sigmoid, lrelu, relu, swish, fastsigmoid, fastswish,
    SGD, ADAM, ADAMAX, NADAM, ADAGRAD, AMSGRAD, ADABOUND,
    softmax, crossentropyloss, mse, onehot

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
