module IceCream

    using LinearAlgebra
    using Distributions
    using StatsBase
    using ReverseDiff
    using CSV
    using Plots

    export icecream, layers, compile, train!, dense,
        sigmoid, lrelu, relu, swish, fastsigmoid, fastswish,
        SGD, ADAM, ADAMAX, NADAM, ADAGRAD, AMSGRAD, ADABOUND,
        softmax, crossentropy, binarycrossentropy, onehot,
        log_cos, mse, mae, pseudo_huber,
        TitanicDataSet

    include("core.jl")
    include("activations.jl")
    include("loss.jl")
    include("optimizer.jl")
    include("metrics.jl")
    include("example.jl")

    using LinearAlgebra
    using Distributions
    using StatsBase
    using ReverseDiff
    using CSV
    using Plots

end
