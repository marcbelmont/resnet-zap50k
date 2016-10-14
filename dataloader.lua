require 'hdf5'
require "csvigo"
local t = require 'transforms'

local M = {}
local DataLoader = torch.class('resnet.DataLoader', M)

function DataLoader.create(batchSize, clOpt)
    local train = M.DataLoader(batchSize, "training", clOpt)
    local validation = M.DataLoader(batchSize, "validation", clOpt)
    return train, validation
end

function DataLoader:__init(batchSize, name, clOpt)
    self.dataset, self.classes, self.inputSize, self.preprocess = DataLoader[clOpt.dataset](name, clOpt)
    self.split = name
    self.__size = self.dataset.targets:size(1)
    self.batchSize = batchSize
end

function DataLoader:size()
    return math.ceil(self.__size / self.batchSize)
end

function DataLoader:run(criterions, simpleMode)
    local perm
    if simpleMode == 1 then
        perm = torch.range(1, self.__size)
    else
        perm = torch.randperm(self.__size)
    end
    local sample = nil
    local n = 0
    local size = self.__size
    local idx = 1
    local function loop()
        if idx > size then
            return nil
        end
        local indices = perm:narrow(1, idx, math.min(self.batchSize, size - idx + 1))
        local inputs = torch.Tensor(indices:size(1), unpack(self.inputSize))
        local targets
        if criterions == 1 then
            targets = torch.Tensor(indices:size(1))
        else
            targets = torch.Tensor(indices:size(1), criterions)
        end
        for i, idx in ipairs(indices:totable()) do
            inputs[i]:copy(self.preprocess(self.dataset.inputs[idx]:float()))
            if criterions == 1 then
                targets[i] = self.dataset.targets[idx][1]
            else
                targets[i]:copy(self.dataset.targets[idx])
            end
            -- image.save(string.format("/tmp/torch/image%s.jpg", idx), inputs[i][1]:mul(85.6):add(191) / 255)
        end
        collectgarbage()
        idx = idx + self.batchSize
        sample = {input = inputs,
                  target = targets}
        n = n + 1
        return n, sample
    end

   return loop
end

-------------
-- Helpers --
-------------

local function load(path)
    local myFile = hdf5.open(path, 'r')
    local function x()
        return {targets = myFile:read("targets"):all(),
                inputs = myFile:read("inputs"):all()}
    end
    local status, data = pcall(x)
    myFile:close()
    if status then
        return data
    end
    return false
end

local function save(path, data)
    local chunkSize = 1024
    local myFile = hdf5.open(path, 'w')
    local options = hdf5.DataSetOptions()
    options:setChunked(chunkSize)
    options:setDeflate(1)
    myFile:write("targets", data.targets, options)

    local options = hdf5.DataSetOptions()
    options:setChunked(chunkSize, chunkSize, chunkSize, chunkSize)
    options:setDeflate(1)
    myFile:write("inputs", data.inputs, options)
    myFile:close()
end

--------------
-- Datasets --
--------------

function DataLoader.shoes2(name, clOpt)
    local width = 100
    local height = 75
    local grayscale = false
    local dimensions = {grayscale and 1 or 3, height, width}
    local backup = string.format("cache/multi-%s.t7", name)
    if paths.filep(backup) and clOpt.debug == 1 then
        return unpack(torch.load(backup))
    end

    -- Read csv, copy targets
    local csvf = csvigo.File(string.format("data/multi-%s.csv", name), "r")
    local row = csvf:read()
    local targets = {}
    local paths = {}
    while row do
        targets[#targets + 1] = {row[2], row[3]}
        paths[#paths + 1] = row[1]
        row = csvf:read()
    end
    csvf:close()
    targets = torch.Tensor(targets)

    -- Copy inputs
    local classes = torch.totable(targets:max(1))[1]
    local inputs = torch.ByteTensor(targets:size(1), grayscale and 1 or 3, height, width)
    for i = 1, #paths do
        local img = image.load(paths[i], 3, "byte")
        if grayscale then
            inputs[i][1] = image.rgb2y(image.scale(img, width, height))
        else
            inputs[i] = image.scale(img, width, height)
        end
    end

    -- Set image pre-processing
    local means, stds = {195, 189, 186}, {82, 87, 89}
    if grayscale then
        means, stds = {191}, {85.6}
    end

    if false then
        for i = 1, grayscale and 1 or 3 do
            means[i] = torch.mean(inputs[{{}, i, {}, {}}]:float())
            stds[i] = torch.std(inputs[{{}, i, {}, {}}]:float())
        end
    end
    local preprocess = t.ColorNormalize({mean = means, std  = stds})
    if name == "training" and clOpt.confidence == 0 then
        preprocess = t.Compose{
            t.ColorNormalize({mean = means, std  = stds}),
            t.SimpleCrop(0),
            t.Rotation(20),
        }
    end

    local result = {{inputs = inputs, targets = targets, paths = paths}, classes, dimensions, preprocess}
    torch.save(backup, result)
    return unpack(result)
end

return M.DataLoader
