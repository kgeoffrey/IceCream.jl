### tesing + loading some data here

using CSV
using StatsBase
using Plots


################################################################################

using IceCream
using Plots

X_train, y_train, X_test, y_test = IceCream.TitanicDataSet()

model = icecream()

structure = layers(
    dense(4, fastsigmoid),
    dense(3, swish),
    dense(1, sigmoid))

compile(model,
    layers = structure,
    X_train = X_train,
    Y_train = y_train,
    loss = binarycrossentropy,
    batchsize = 100)

@time train!(model, optimizer = NADAM, α = 0.01, epochs = 2000)
plot(model.model_loss)
