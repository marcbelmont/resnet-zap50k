local Models = {}

local Convolution = nn.SpatialConvolution
local Avg = nn.SpatialAveragePooling
local ReLU = nn.ReLU
local Max = nn.SpatialMaxPooling
local SBatchNorm = nn.SpatialBatchNormalization

-- The shortcut layer is either identity or 1x1 convolution
local function shortcut(nInputPlane, nOutputPlane, stride)
    if nInputPlane ~= nOutputPlane then
        -- 1x1 convolution
        return nn.Sequential()
            :add(Convolution(nInputPlane, nOutputPlane, 1, 1, stride, stride))
            :add(SBatchNorm(nOutputPlane))
    else
        return nn.Identity()
    end
end

local function BNInit(model, name)
    for k, v in pairs(model:findModules(name)) do
        v.weight:fill(1)
        v.bias:zero()
    end
end

local function ConvInit(model, name)
    for k, v in pairs(model:findModules(name)) do
        local n = v.kW * v.kH * v.nOutputPlane
        v.weight:normal(0, math.sqrt(2 / n))
        v.bias:zero()
    end
end

-- The basic residual layer block for 18 and 34 layer network, and the
-- CIFAR networks
local function basicBlock(nInputPlane, nOutputPlane, stride)
    local s = nn.Sequential()
    s:add(nn.SpatialConvolution(nInputPlane, nOutputPlane, 3, 3, stride, stride, 1, 1))
    s:add(nn.SpatialBatchNormalization(nOutputPlane))
    s:add(nn.ReLU(true))
    s:add(nn.SpatialConvolution(nOutputPlane, nOutputPlane, 3, 3, 1, 1, 1, 1))
    s:add(nn.SpatialBatchNormalization(nOutputPlane))

    return nn.Sequential()
        :add(nn.ConcatTable()
                 :add(s)
                 :add(shortcut(nInputPlane, nOutputPlane, stride)))
        :add(nn.CAddTable(true))
        :add(ReLU(true))
end

function Models.CNN(sizes, classes)
    local input = nn.Identity()()
    local net = input
    local seq = nn.Sequential()
    local inputPlanes = sizes[1]

    -- convolution layers
    local outputPlanes = 16
    seq:add(nn.SpatialConvolution(inputPlanes, outputPlanes, 7, 7, 2, 2, 3, 3))
    inputPlanes = outputPlanes
    seq:add(nn.SpatialBatchNormalization(outputPlanes))
    seq:add(nn.ReLU(true))

    -- residual layers
    for _, outputPlanes in pairs({16, 16, 16, 32, 32, 32, 64, 64, 64}) do
        seq:add(basicBlock(inputPlanes,
                           outputPlanes,
                           inputPlanes == outputPlanes and 1 or 2))
        inputPlanes = outputPlanes
    end
    ConvInit(seq, 'nn.SpatialConvolution')
    BNInit(seq, 'nn.SpatialBatchNormalization')

    -- fully connected
    seq:add(nn.SpatialAveragePooling(13, 10))
    size = inputPlanes
    seq:add(nn.View(size))
    net = seq(net)

    if #classes == 2 then
        seq1 = nn.Sequential()
        seq1:add(nn.Linear(size, classes[1]))
        seq1:add(nn.BatchNormalization(classes[1]))
        seq1:add(nn.LogSoftMax())

        seq2 = nn.Sequential()
        seq2:add(nn.Linear(size, classes[2]))
        seq2:add(nn.BatchNormalization(classes[2]))
        seq2:add(nn.LogSoftMax())

        pll = nn.ConcatTable()
        pll:add(seq1)
        pll:add(seq2)
        net = pll(net)
    else
        seq = nn.Sequential()
        seq:add(nn.Linear(size, classes))
        seq:add(nn.BatchNormalization(classes))
        seq:add(nn.LogSoftMax())
        net = seq(net)
    end
    return nn.gModule({input}, {net})
end

return Models
