### tesing + loading some data here

using CSV
using StatsBase
using Plots

training = CSV.read("src/titanic_data/train.csv")
test = CSV.read("src/titanic_data/test.csv")

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
    loss = crossentropy,
    batchsize = 100)

train!(model, optimizer = NADAM, α = 0.001, epochs = 20)
plot(model.model_loss)
