module Icecream

export icecream, layers, compile, train!, dense,
    sigmoid, lrelu, relu, swish, fastsigmoid, fastswish,
    SGD, ADAM, ADAMAX, NADAM, ADAGRAD, AMSGRAD, ADABOUND,
    softmax, crossentropyloss, onehot,
    log_cos, mse, mae, pseudo_huber

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
