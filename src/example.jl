
## Load Titanic Data set for example

function TitanicDataSet()
    f1 = joinpath(dirname(pathof(IceCream)), "titanic_data", "train.csv")
    f2 = joinpath(dirname(pathof(IceCream)), "titanic_data", "test.csv")


    training = CSV.read(f1)
    test = CSV.read(f2)

    # joinpath(dirname(pathof(MyPkg)), "..", "data")

    y_train = training.Survived
    X_train = training[setdiff(names(training), [:Survived])]
    y_test = test.Survived
    X_test = test[setdiff(names(test), [:Survived])]

    y_test = convert(Array{Float64}, y_test)
    y_train = convert(Array{Float64}, y_train)

    X_test = convert(Array{Float64}, X_test)
    X_train = convert(Array{Float64}, X_train)

    return X_train, y_train, X_test, y_test
end
