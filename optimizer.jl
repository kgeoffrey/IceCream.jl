function update!(obj::icecream)
    obj.idx .= sample(1:size(obj.X_train,1), obj.batchsize)
    obj.X_sample .= view(obj.X_train, obj.idx,:)
    obj.Y_sample .= view(obj.Y_train, obj.idx,:)
    gradients = ReverseDiff.gradient!(obj.pre_arrays, obj.tape, (obj.X_sample, obj.Y_sample, obj.weights...))
    obj.gradients = gradients[3:end]
    nothing
end

function metrics!(obj::icecream)
    append!(obj.model_loss, obj.loss_function(obj.X_train, obj.Y_train , obj.weights...))
    nothing
end


function SGD(obj::icecream,alpha::Float64, epochs::Int)
    @inbounds for _ in 1:epochs
        @inbounds for _ in 1:1
            update!(obj)
            for i in 1:length(obj.gradients)
                obj.weights[i] .-= (alpha * obj.gradients[i])
            end
        end
        metrics!(obj)
    end
end

function ADAM(obj::icecream, alpha::Float64, epochs::Int, beta1 = 0.9, beta2 = 0.999)
    t = 1
    m, v, m_hat, v_hat = initializer(obj, 4)
    @inbounds for _ in 1:epochs
        update!(obj)
        for i in 1:length(obj.gradients)
            m[i] .= beta1 .* m[i] .+ (1 .- beta1) .* obj.gradients[i]
            v[i] .= beta2 .* v[i] .+ (1 .- beta2) .* obj.gradients[i] .^ 2
            m_hat[i] .= m[i] ./ (1 .- beta1 .^ t)
            v_hat[i] .= v[i] ./ (1 .- beta2 .^ t)
            obj.weights[i] .-= alpha .* m_hat[i] ./ (sqrt.(v_hat[i]) .+ eps(1.0))
            t += 1
        end
        metrics!(obj)
    end
    nothing
end

function ADAMAX(obj::icecream, alpha::Float64, epochs::Int, beta1 = 0.9, beta2 = 0.999)
    t = 1
    m, v, m_hat, v_hat = initializer(obj, 4)
    @inbounds for _ in 1:epochs
        update!(obj)
        for i in 1:length(obj.gradients)
            m[i] .= beta1 .* m[i] .+ (1 .- beta1) .* obj.gradients[i]
            m_hat[i] .= m[i] ./ (1 .- beta1 .^ t)
            v[i] .= max.(beta2 .* v[i], abs.(obj.gradients[i]))
            obj.weights[i] .-= alpha .* m_hat[i] ./ (v[i] .+ eps(1.0))
            t += 1
        end
        metrics!(obj)
    end
    nothing
end

function NADAM(obj::icecream, alpha::Float64, epochs::Int, beta1 = 0.9, beta2 = 0.999)
    t = 1
    m, v, m_hat, v_hat = initializer(obj, 4)
    @inbounds for _ in 1:epochs
        update!(obj)
        for i in 1:length(obj.gradients)
            m[i] .= beta1 .* m[i] .+ (1 .- beta1) .* obj.gradients[i]
            v[i] .= beta2 .* v[i] .+ (1 .- beta2) .* obj.gradients[i] .^ 2
            m_hat[i] .= m[i] ./ (1 .- beta1 .^ t) .+ (1 .- beta1) .* obj.gradients[i] ./ (1 - (beta1).^t) 
            v_hat[i] .= v[i] ./ (1 .- beta2 .^ t)
            obj.weights[i] .-= alpha .* m_hat[i] ./ (sqrt.(v_hat[i]) .+ eps(1.0))
            t += 1
        end
        metrics!(obj)
    end
    nothing
end
