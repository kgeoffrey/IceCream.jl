### core

## icecream model

mutable struct icecream
    X_train::AbstractArray
    Y_train::AbstractArray
    layers::Tuple
    weights::Tuple
    feedforward::Function
    loss_function::Function
    loss::Function
    idx::AbstractArray
    X_sample::AbstractArray
    Y_sample::AbstractArray
    tape
    pre_arrays
    optimizer::Function
    batchsize::Int
    model_loss::AbstractArray
    icecream() = new()
end


######################
### static methods ###
######################

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

function layers(structure::Tuple...)
    tuple(structure...)
end

function addbias(x::AbstractArray)
    b = hcat(ones(size(x,1)), x)
    return b
end


########################
### icecream methods ###
########################

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

function initialize_model(obj::icecream)
    forwardpass = (X, w...) -> feedforward(X, obj.layers, w...)
    obj.feedforward = forwardpass
    lossmodel = (X, Y, w...) -> obj.loss(forwardpass(X, w...), Y)
    obj.loss_function = lossmodel
    nothing
end

## icecream call

function compile(obj::icecream, ;X_train::AbstractArray, Y_train::AbstractArray, loss::Function, batchsize::Int, layers::Tuple)
    obj.layers = layers
    obj.X_train = X_train
    obj.Y_train = Y_train
    obj.batchsize = batchsize
    obj.idx = sample(1:size(obj.X_train,1), obj.batchsize)
    obj.X_sample = view(obj.X_train,obj.idx,:)
    obj.Y_sample = view(obj.Y_train,obj.idx,:)
    obj.loss = loss

    initialize_weights(obj)
    initialize_model(obj)
    initialize_tape(obj)
    nothing
end

function train(obj::icecream,;optimizer::Function, alpha::Float64, epochs::Int)
    obj.optimizer = optimizer
    optimizer(obj, alpha, epochs)
end
