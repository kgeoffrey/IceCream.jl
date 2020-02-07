### metrics

## for classification
function accuracy(pred, y)
    s = [if i > 0.5 1 else 0 end for i in pred]
    missclass = sum(abs.(s - onehot(y)))/size(s,2)
    return (size(s,1) - missclass)/size(s,1)
end
