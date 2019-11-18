### old stuff

### testing diffferent data types
using Pkg
using LinearAlgebra
using Plots
using DataFrames, CSV
using StatsBase
using DataFrames
using Distributions
using ReverseDiff
using MLDatasets

sigmoid(x::AbstractArray) = 1 ./ (1 .+ exp.(.-x))
relu(x::AbstractArray) = max.(zero(x),x)
lrelu(x::AbstractArray) = max.(0.01*x, x)
swish(x::AbstractArray) = x .* sigmoid(x)

training = CSV.read("titanic/julia_data/train.csv")
test = CSV.read("titanic/julia_data/test.csv")

y_train = training.Survived
x_train = training[setdiff(names(training), [:Survived])]
y_test = test.Survived
x_test = test[setdiff(names(test), [:Survived])]

Y_test = convert(Array, y_test)
Y_train = convert(Array, y_train)
## preparing X matrix for regression
X = convert(Matrix, x_train)
X_trans = fit(UnitRangeTransform, X)
xx = StatsBase.transform(X_trans,X)
#newx = hcat(ones(size(xx,1)), xx)
newx = xx

y_train = convert(Array{Float64}, y_train)

nothing

sigmoid(x::AbstractArray) = 1 ./ (1 .+ exp.(.-x))
relu(x::AbstractArray) = max.(zero(x),x)
lrelu(x::AbstractArray) = max.(0.01*x, x)
swish(x::AbstractArray) = x .* sigmoid(x)

const ε = eps(1.0)
crossentropyloss(x::AbstractArray, y::AbstractArray) = - sum(y.*log.((x).+ ε))/size(x,1)
mse(x::AbstractArray, y::AbstractArray) = mean((y .- x).^2)

mutable struct icecream
    ## General Data
    X_train::AbstractArray
    Y_train::AbstractArray
    layers::Tuple
    weights::Tuple

    ## storing compiled model
    feedforward::Function
    loss_function::Function
    loss::Function
    idx::AbstractArray
    X_sample::AbstractArray
    Y_sample::AbstractArray
    tape
    pre_arrays

    ## storing data for training
    optimizer::Function
    batchsize::Int
    model_loss::AbstractArray

    icecream() = new()
end


### static methods

function dense(nodes::Int, Activation::Function)
    return (nodes,Activation)
end

function feedforward(X, structure, w...)
    layer = structure[1][2](addbias(X) * w[1])
    for i in 2:length(structure)
        layer = structure[i][2](addbias(layer) * w[i])
    end
    return layer
end


# function layers(structure::Tuple)
#     tuple(structure)
#     nothing
# end

function layers(structure::Tuple...)
    tuple(structure...)
end

function addbias(x::AbstractArray)
    b = hcat(ones(size(x,1)), x)
    return b
end

### Type methods for functions

function initialize_weights(obj::icecream)
    distribution = Normal(.5, 1)
    weights = tuple(rand(distribution, obj.layers[1][1], size(obj.X_sample,2)+1)')
    for i in 1:length(structure)-1
        w_matrix = rand(distribution, obj.layers[i+1][1], obj.layers[i][1]+1)
        weights = tuple(weights..., w_matrix')
    end
    obj.weights = weights
    nothing
end

function initialize_tape(obj::icecream)

    pre_arrays = (obj.X_sample, obj.Y_sample, obj.weights...)
    allocation = tuple()
    for i in 1:length(pre_arrays)
        allocation = (allocation..., similar(pre_arrays[i]))
    end

    model.pre_arrays = allocation

    tape = ReverseDiff.compile(ReverseDiff.GradientTape(obj.loss_function, (obj.X_sample, obj.Y_sample, obj.weights...)))
    obj.tape = tape
    nothing
end


function compile(obj::icecream, ;X_train::AbstractArray, Y_train::AbstractArray, loss::Function, batchsize::Int, layers::Tuple)
    ## Initializing samples for batch training
    obj.layers = layers
    obj.X_train = X_train
    obj.Y_train = Y_train
    obj.batchsize = batchsize
    obj.idx = sample(1:size(obj.X_train,1), obj.batchsize)
    obj.X_sample = view(obj.X_train,obj.idx,:)
    obj.Y_sample = view(obj.Y_train,obj.idx,:)
    obj.loss = loss

    # Initializing weights
    initialize_weights(obj)

    initialize_model(obj)

    # Initializing differentiable function
    #forwardpass = (X, w...) -> feedforward(X, obj.layers, w...)
    #obj.feedforward = forwardpass
    #lossmodel = (X, Y, w...) -> loss(forwardpass(X, w...), Y)
    #obj.model_loss = lossmodel
    # prerecording tape + allocating arrays
    initialize_tape(obj)

    nothing
end

function train(obj::icecream,;optimizer::Function, alpha::Float64, epochs::Int)
    obj.optimizer = optimizer
    optimizer(obj, alpha, epochs)
end

function initialize_model(obj::icecream)
    forwardpass = (X, w...) -> feedforward(X, obj.layers, w...)
    obj.feedforward = forwardpass
    lossmodel = (X, Y, w...) -> obj.loss(forwardpass(X, w...), Y)
    obj.loss_function = lossmodel
    nothing
end

function SGD(obj::icecream,alpha::Float64, epochs::Int)
    losses =[]
    @inbounds for _ in 1:epochs
        @inbounds for _ in 1:1
            obj.idx .= sample(1:size(obj.X_train,1), obj.batchsize)
            obj.X_sample .= view(obj.X_train, obj.idx,:)
            obj.Y_sample .= view(obj.Y_train, obj.idx,:)
            gradients = ReverseDiff.gradient!(obj.pre_arrays, obj.tape, (obj.X_sample, obj.Y_sample, obj.weights...))
            weight_gradients = gradients[3:end]
            for i in 1:length(weight_gradients)
                obj.weights[i] .-= (alpha * weight_gradients[i])
            end
        end

        append!(losses, obj.loss_function(obj.X_train, obj.Y_train , obj.weights...))
    end
    obj.model_loss = losses
    nothing
end

model = icecream()

structure = layers(
    dense(4, sigmoid),
    dense(3, sigmoid),
    dense(1,sigmoid))

compile(model,
    layers = structure,
    X_train = newx,
    Y_train = y_train,
    loss = mse,
    batchsize = 10)

@time train(model, optimizer = SGD, alpha = 0.1, epochs = 2000)
