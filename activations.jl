### activations ###

sigmoid(x::AbstractArray) = 1 ./ (1 .+ exp.(.-x))
relu(x::AbstractArray) = max.(zero(x),x)
lrelu(x::AbstractArray) = max.(0.01*x, x)
swish(x::AbstractArray) = x .* sigmoid(x)
const alph = 0.1
fastsigmoid(x::AbstractArray) = 0.5 .* (x .* alph ./ (1 .+ abs.(x .* alph))) .+ 0.5

const bet = 1.5
fastswish(x::AbstractArray) = x .* (0.5 .* (x .* bet ./ (1 .+ abs.(x .* bet))) .+ 0.5)

