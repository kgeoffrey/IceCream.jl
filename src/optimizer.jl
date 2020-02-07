### mutable functions

function update!(obj::icecream)
    obj.idx .= sample(1:size(obj.X_train,1), obj.batchsize)
    obj.X_sample .= view(obj.X_train, obj.idx,:)
    obj.Y_sample .= view(obj.Y_train, obj.idx,:)
    gradients = ReverseDiff.gradient!(obj.pre_arrays, obj.tape, (obj.X_sample, obj.Y_sample, obj.weights...))
    obj.∇ = gradients[3:end]
    nothing
end

function metrics!(obj::icecream)
    append!(obj.model_loss, obj.loss_function(obj.X_train, obj.Y_train , obj.weights...))
    nothing
end

### optimizers

function SGD(obj::icecream, α::Float64, epochs::Int)
    @inbounds for _ in 1:epochs
        @inbounds for _ in 1:1
            update!(obj)
            for i in 1:length(obj.∇)
                obj.weights[i] .-= (α * obj.∇[i])
            end
        end
        metrics!(obj)
    end
end

function ADAM(obj::icecream, α::Float64, epochs::Int, β₁ = 0.9, β₂ = 0.999)
    t = 1
    𝕄, 𝕍, 𝒎, 𝒗 = initializer(obj, 4)
    @inbounds for _ in 1:epochs
        update!(obj)
        for i in 1:length(obj.∇)
            𝕄[i] .= β₁ .* 𝕄[i] .+ (1 .- β₁) .* obj.∇[i]
            𝕍[i] .= β₂ .* 𝕍[i] .+ (1 .- β₂) .* obj.∇[i] .^ 2
            𝒎[i] .= 𝕄[i] ./ (1 .- β₁ .^ t)
            𝒗[i] .= 𝕍[i] ./ (1 .- β₂ .^ t)
            obj.weights[i] .-= α .* 𝒎[i] ./ (sqrt.(𝒗[i]) .+ eps(1.0))
            t += 1
        end
        metrics!(obj)
    end
end

function ADAMAX(obj::icecream, α::Float64, epochs::Int, β₁ = 0.9, β₂ = 0.999)
    t = 1
    𝕄, 𝕍, 𝒎 = initializer(obj, 3)
    @inbounds for _ in 1:epochs
        update!(obj)
        for i in 1:length(obj.∇)
            𝕄[i] .= β₁ .* 𝕄[i] .+ (1 .- β₁) .* obj.∇[i]
            𝒎[i] .= 𝕄[i] ./ (1 .- β₁ .^ t)
            𝕍[i] .= max.(β₂ .* 𝕍[i], abs.(obj.∇[i]))

            obj.weights[i] .-= α .* 𝒎[i] ./ (𝕍[i] .+ eps(1.0))
            t += 1
        end
        metrics!(obj)
    end
end



function NADAM(obj::icecream, α::Float64, epochs::Int, β₁ = 0.9, β₂ = 0.999)
    t = 1
    𝕄, 𝕍, 𝒎, 𝒗 = initializer(obj, 4)
    @inbounds for _ in 1:epochs
        update!(obj)
        for i in 1:length(obj.∇)
            𝕄[i] .= β₁ .* 𝕄[i] .+ (1 .- β₁) .* obj.∇[i]
            𝕍[i] .= β₂ .* 𝕍[i] .+ (1 .- β₂) .* obj.∇[i] .^ 2
            𝒎[i] .= 𝕄[i] ./ (1 .- β₁ .^ t) .+ (1 .- β₁) .* obj.∇[i] ./ (1 - (β₁).^t)
            𝒗[i] .= 𝕍[i] ./ (1 .- β₂ .^ t)

            obj.weights[i] .-= α .* 𝒎[i] ./ (sqrt.(𝒗[i]) .+ eps(1.0))
            t += 1
        end
        metrics!(obj)
    end
end

function ADAGRAD(obj::icecream, α::Float64, epochs::Int, β₁ = 0.9, β₂ = 0.999)
    t = 1
    𝕄, 𝕍 = initializer(obj, 4)
    @inbounds for _ in 1:epochs
        update!(obj)
        for i in 1:length(obj.∇)
            𝕄[i] .= 𝕄[i] .+ obj.∇[i] .^2

            obj.weights[i] .-= α ./ (sqrt.(𝕄[i]) .+ eps(1.0)) .* obj.∇[i]
            t += 1
        end
        metrics!(obj)
    end
end


function AMSGRAD(obj::icecream, α::Float64, epochs::Int, β₁ = 0.9, β₂ = 0.999)
    t = 1
    𝕄, 𝕍, 𝒗 = initializer(obj, 3)
    @inbounds for _ in 1:epochs
        update!(obj)
        for i in 1:length(obj.∇)
            𝕄[i] .= β₁ .* 𝕄[i] .+ (1 .- β₁) .* obj.∇[i]
            τ = 𝕍[i]
            𝕍[i] .= β₂ .* 𝕍[i] .+ (1 .- β₂) .* obj.∇[i] .^ 2
            𝒗[i] .= max.(𝕍[i], τ)

            obj.weights[i] .-= α ./ (sqrt.(𝒗[i]) .+ eps(1.0)) .* obj.∇[i]
            t += 1
        end
        metrics!(obj)
    end
end

function ADABOUND(obj::icecream, α::Float64, epochs::Int, β₁ = 0.9, β₂ = 0.999)
    t = 1
    𝕄, 𝕍, 𝒎, 𝒗, τ = initializer(obj, 5)
    ηl(t) = 0.1 .- 0.1 ./ ((1 .- β₂).*(t+1))
    ηu(t) = 0.1 .+ 0.1 ./ ((1 .- β₂).*t)

    function clip(X, l, u)
        n = norm(X)
        if n >= u
            return (u/n) .* X
        elseif n <= l
            return (l/n) .* X
        else
            return X
        end
    end

    @inbounds for _ in 1:epochs
        update!(obj)
        for i in 1:length(obj.∇)
            β₁ = β₁/t
            𝕄[i] .= β₁ .* 𝕄[i] .+ (1 .- β₁) .* obj.∇[i]
            𝕍[i] .= β₂ .* 𝕍[i] .+ (1 .- β₂) .* obj.∇[i] .^ 2

            𝒎[i] .= 𝕄[i] ./ (1 .- β₁ .^ t) .+ (1 .- β₁) .* obj.∇[i] ./ (1 - (β₁).^t)
            𝒗[i] .= 𝕍[i] ./ (1 .- β₂ .^ t)

            τ[i] .= clip(α ./ sqrt.(𝕍[i]), ηl(t), ηu(t)) ./ sqrt(t)

            obj.weights[i] .-= τ[i] .* 𝕄[i]
            t += 1
        end
        metrics!(obj)
    end
end
