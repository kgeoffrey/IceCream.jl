"""
Should be all the different types of layers... for now only dense and softmax.
need:
- Recurrent Neural nets
- Conv Neural nets
- LSTM nets

for now this file is not connected or used in icecream module
"""

function dense(nodes::Int, Activation::Function)
    return (nodes,Activation)
end

function softmax(xs::AbstractArray; dims=2) ## this should be in layers
    max_ = maximum(xs, dims=dims)
    exp_ = exp.(xs .- max_)
    exp_ ./ sum(exp_, dims=dims)
end

### Helper functions ###

function addbias(x::AbstractArray)
    b = hcat(ones(size(x,1)), x)
    return b
end
