### tesing + loading some data here

using CSV
using StatsBase
using Plots

training = CSV.read("julia_data/train.csv")
test = CSV.read("julia_data/test.csv")

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
newx = xx
y_train = convert(Array{Float64}, y_train)


################################################################################


include("./Icecream.jl")
using .Icecream


model = icecream()

structure = layers(
    dense(4, sigmoid),
    dense(3, sigmoid),
    dense(1, sigmoid))

compile(model,
    layers = structure,
    X_train = newx,
    Y_train = y_train,
    loss = mae,
    batchsize = 10)

@time train!(model, optimizer = NADAM, α = 0.01, epochs = 2000)
plot(model.model_loss)
