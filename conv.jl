#### testing conv filters and maxpooling

t = rand(5, 8)

filter = rand(3,3)

mutable struct convolution
    idx::Vector
    subimages::Vector
    filter
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



## convolution function
function conv(img, filters, numfilters)
    f = size(filters,1)-1
    h, w = size(img)
    image = Array{Float64}(undef,h-f, w-f)

    imgs, idx = split_img(img, filters)

    for i in 1:length(imgs)
        image[idx[i]...] = sum(imgs[i] .* filters)
    end

    return image
end

function filters(size, number)
    filterlist = []

    for i in 1:number
        push!(filterlist, rand(size, size))
    end
    return filterlist
end
### to do: multiple filters per image!

conv(t, filter, 2)
imgs, idx = split_img(t, filter)
sum(imgs[1] .* filter)

filters(3, 5)
