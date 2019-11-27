### functionality I want to add


### compare method ### example comparing different optimizers 

function compare(obj::icecream,;optimizers::Tuple, α::Float64, epochs::Int)
    metrics = Array{Float64}(undef, epochs, 0)

    for i in optimizers
        new = icecream()

        structure = layers(
            dense(10, fastswish),
            dense(5, fastswish),
            dense(2, fastswish),
            dense(1,fastsigmoid))

        compile(new,
            layers = structure,
            X_train = newx,
            Y_train = y_train,
            loss = mse,
            batchsize = 60)

        train!(new, optimizer = i, α = α, epochs = epochs)
        metrics = hcat(metrics, new.model_loss)
    end

    metrics = convert(Array{Float64, 2}, metrics)

    #plot(metrics, label = map(string, optimizers))

    p = plot()
    for i in 1:length(optimizers)
        plot!(p, metrics[:,i], label = string(optimizers[i]))
    end

    return p
end

compare(new,
    optimizers = (ADAM, NADAM, ADAMAX, AMSGRAD),
    α = 0.01,
    epochs = 1000)
