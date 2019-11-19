function softmax(xs::AbstractArray; dims=2) ## this should be in layers
    max_ = maximum(xs, dims=dims)
    exp_ = exp.(xs .- max_)
    exp_ ./ sum(exp_, dims=dims)
end

## classification
const ε = eps(1.0)
crossentropy(x::AbstractArray, y::AbstractArray) = - sum(y.*log.((x).+ ε))/size(x,1)
m = x -> length(x)
binary_crossentropy(x::AbstractArray, y::AbstractArray) = -(1/m(y))*(transpose(y)*log.(x .+ ε) + transpose(ones(m(y))-y)*log.(ones(m(y))-(x .- ε)))

function onehot(y::Array)
    class = [convert(Array{Float64},y.== c) for c in unique(y)]
    class = hcat(class...)
    return class
end


## regression
mse(x::AbstractArray, y::AbstractArray) = mean((y .- x).^2)
mae(x::AbstractArray, y::AbstractArray) = mean(abs.(y .- x))
log_cos(x::AbstractArray, y::AbstractArray) = sum(log.(cosh.(x .- y)))
pseudo_huber(x::AbstractArray, y::AbstractArray, ;δ::Float64 = 1) = δ.^2 .* (sqrt.(1 .+ ((x .- y)./δ).^2 .- 1)
