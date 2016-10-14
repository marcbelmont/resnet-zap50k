require "torch"
require "image"
require "pprint"
require "nngraph"
require "optim"
local curl = require "cURL"
local gnuplot = require "gnuplot"
local manifold = require "manifold"
local cjson = require "cjson"

local Trainer = require "train"
local DataLoader = require "dataloader"
local Models = require "model"

-------------------
-- Configuration --
-------------------

local MAX_EPOCHS = 200
local CHECKPOINT_PATH = "output/checkpoint-%s.t7"
local CONFIG_PATH = "output/config-%s.t7"
local FEATURES_PATH = "cache/features-%s.t7"
local FEATURES_EXT_PATH = "cache/features-ext.t7"

local function initConfig(clOpt)
    -- Restore
    if clOpt.restore > 0 then
        return torch.load(string.format(CONFIG_PATH, clOpt.restore))
    end

    -- Init config
    local config = {
        batchSize = clOpt.server == 1 and 128 or 32,
        bestAccuracy = nil,
        tExamples = 0,

        -- Model
        note = "",
        model = "CNN",
        criterions = 2,
        optim = {
            method = "sgd",
            learningRate = .01,
            momentum = .9,
            nesterov = true,
            dampening = 0,
            weightDecay = 1e-4,
    }}

    config.optim.learningRate0 = config.optim.learningRate
    return config
end

local function options()
    local cmd = torch.CmdLine()
    -- modes
    cmd:option("-debug", 0)
    cmd:option("-search", -1)
    cmd:option("-server", 0)
    -- actions
    cmd:option("-dataset", "")
    cmd:option("-restore", 0)
    cmd:option("-features", 0)
    cmd:option("-confidence", 0)
    local opt = cmd:parse(arg or {})
    if opt.search == -1 then
        opt.search = nil
    end
    return opt
end

local function init()
    local clOpt = options()
    local config = initConfig(clOpt)

    -- debug
    if clOpt.search then
        MAX_EPOCHS = 1
    end
    if clOpt.debug == 1 then
        MAX_EPOCHS = 1
        config.batchSize = 4
        nngraph.setDebug(true)
    end

    -- init
    torch.setnumthreads(1) -- faster on CPU
    torch.setdefaulttensortype('torch.FloatTensor') -- faster, less memory
    torch.manualSeed(1)
    return clOpt, config
end

local function tsne(features)
    torch.setdefaulttensortype('torch.DoubleTensor')
    local embedding = manifold.embedding.tsne(features:double(), {dim = 2, perplexity = 30})
    torch.setdefaulttensortype('torch.FloatTensor')
    gnuplot.pngfigure("/tmp/torch/tsne.png")
    gnuplot.raw("set terminal png medium size 1000,800")
    gnuplot.plot(embedding, "+")
    gnuplot.plotflush()
end

----------------------
-- Train & evaluate --
----------------------

local function featModel(model, outputSize)
    -- Remove classifier layer
    model.forwardnodes[4].data.module = nn.Identity()
    return model
end

local function getFeatures(clOpt, config)
    -- Create feature vectors
    local path
    if clOpt.features == 2 then
        path = FEATURES_EXT_PATH
    else
        path = string.format(FEATURES_PATH, clOpt.restore)
    end
    local model = torch.load(string.format(CHECKPOINT_PATH, clOpt.restore))
    local trainer = Trainer(model, config)
    local loader = DataLoader(config.batchSize, "validation", clOpt)
    model = featModel(model, loader.classes)
    local features = trainer:features(loader, clOpt)
    local npy4th = require 'npy4th'
    npy4th.savenpy(path .. ".npy", features.inputs)
    file = io.open(path .. ".json", "w")
    file:write(cjson.encode(features.paths))
    file:close()
end

local function trainAndTest(clOpt, config)
    local trainLoader, valLoader = DataLoader.create(config.batchSize, clOpt)
    local model
    if clOpt.restore == 0 then
        -- Create model
        model = Models[config.model](trainLoader.inputSize, trainLoader.classes)
    else
        model = torch.load(string.format(CHECKPOINT_PATH, clOpt.restore))
        -- Fine tune, set new classifier
        -- model:clearState()
        local seq = model:get(3)
        local outputSize = trainLoader.classes
        local index = 1
        if seq:get(index).weight:size(2) ~= outputSize then
            local inputSize = seq:get(index).weight:size(2)
            seq:remove(index)
            seq:insert(nn.Linear(inputSize, outputSize), index)
        end
    end
    local trainer = Trainer(model, config)

    -- show info
    local info = pprint.string(config):gsub(" ", "") .. ""

    local experimentId = clOpt.restore

    -- train
    print(info, experimentId)
    local bestLoss, bestEpoch
    for epoch = 1, MAX_EPOCHS do
        print("Epoch " .. epoch)

        -- Train
        local trainingLoss, trainingAccuracy, speed = trainer:train(trainLoader, clOpt)

        -- Validation
        local validationLoss, validationAccuracy = trainer:test(valLoader, clOpt)

        -- Decay learningRate
        config.optim.learningRate = config.optim.learningRate0 * (1 - epoch / MAX_EPOCHS)

        -- save best model
        if not config.bestAccuracy or validationAccuracy > config.bestAccuracy then
            print("Saving...")
            config.bestAccuracy = validationAccuracy
            local dfdx = config.optim.dfdx
            config.optim.dfdx = nil
            if clOpt.debug == 0 then
                torch.save(string.format(CONFIG_PATH, experimentId), config)
                torch.save(string.format(CHECKPOINT_PATH, experimentId), model:clearState())
            end
            config.optim.dfdx = dfdx
        end

        -- early stopping
        if not bestLoss or validationLoss < bestLoss then
            bestLoss = validationLoss
            bestEpoch = epoch
        end
        if epoch > 2 * bestEpoch and epoch > 100 then
            print("Early stopping!")
            break
        end
    end
end

---------
-- Run --
---------

local clOpt, config = init()
if clOpt.features > 0 then
    getFeatures(clOpt, config)
else
    trainAndTest(clOpt, config)
end
