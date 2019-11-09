function softmax(xs::AbstractArray; dims=2)
    max_ = maximum(xs, dims=dims)
    exp_ = exp.(xs .- max_)
    exp_ ./ sum(exp_, dims=dims)
end
const ε = eps(1.0)

crossentropyloss(x::AbstractArray, y::AbstractArray) = - sum(y.*log.((x).+ ε))/size(x,1)
mse(x::AbstractArray, y::AbstractArray) = mean((y .- x).^2)


function onehot(y::Array)
    class = [convert(Array{Float64},y.== c) for c in unique(y)]
    class = hcat(class...)
    return class
end

## for classification
function accuracy(pred, y)
    s = [if i > 0.5 1 else 0 end for i in pred]
    missclass = sum(abs.(s - onehot(y)))/size(s,2)
    return (size(s,1) - missclass)/size(s,1)
end
