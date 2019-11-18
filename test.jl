### tesing

include("./Icecream.jl")
import Icecream


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
