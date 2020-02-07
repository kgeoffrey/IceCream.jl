#### testing conv filters and maxpooling

t = rand(10, 10)

using ReverseDiff
using LinearAlgebra

mutable struct conv
    idx::Vector
    subimages::Vector
    filters::Vector
    image::Array

    function convolution()
    convolution() = new()
end

c = convolution()


function split_img(t, filter)
    imgs =[]
    idxlist = []
    h, w = size(t)
    f = size(filter,1)-1
    for i in 1:h-f
        for j in 1:w-f
            img = view(t,i:i+f, j:j+f)
            push!(imgs, img)
            push!(idxlist, (i,j))
        end
    end
    return imgs, idxlist
end


function get_filters(size, number)
    filterlist = []
    for i in 1:number
        push!(filterlist, rand(size, size))
    end
    return filterlist
end

## convolution function
function conv(img, filters) #filters

    # filters = get_filters(f_size, f_number)
    #f = f_size-1
    f_number = length(filters[1][1])
    f = size(filters,1) -1
    h, w = size(img)
    image = Array{Float64}(undef,h-f, w-f, f_number)

    for j in 1:f_number
        imgs, idx = split_img(img, filters)
        for i in 1:length(imgs)
            image[idx[i]..., j] += sum(imgs[i] .* filters)
            print()
        end
    end
    return image
end



### to do: multiple filters per image!


@time conv(t, kern)
convo = x -> (conv(t, x))

filters = get_filters(3, 2)

@time convo(filters[1])

kern = [1. 0. 1.; 2. 0. 2.; 1. 0. 1.]


ReverseDiff.gradient(convo, kern)

## investigate
function meme(x)
    this = rand(3,3)
    image = zeros(3,3)
    for i in 1:length(image)
        image[i] = sum(this .* x)
    end
    return (image)
end



convo = x -> norm(conv(t, x))

hm = ReverseDiff.gradient(meme, kern)
