require "ParallelCriterionSkip"
local DataLoader = require "dataloader"

local M = {}

local Trainer = torch.class('resnet.Trainer', M)

function Trainer:__init(model, config)
    self.model = model
    self.criterions = config.criterions or 1
    if config.criterions and config.criterions > 1 then
        nl1 = nn.ClassNLLCriterion()
        nl2 = nn.ClassNLLCriterion()
        self.criterion = nn.ParallelCriterionSkip():add(nl1):add(nl2, .5)
    else
        self.criterion = nn.ClassNLLCriterion()
    end
    self.optim = optim[config.optim.method]
    self.optimState = config.optim
    self.params, self.gradParams = self.model:getParameters()
end

function Trainer:copyInputs(sample)
    self.input = self.input or torch.Tensor()
    self.input:resize(sample.input:size()):copy(sample.input)
    self.target = self.target or torch.Tensor()
    self.target:resize(sample.target:size()):copy(sample.target)
end

-----------
-- Train --
-----------

function Trainer:train(dataloader, clOpt)
    self.model:training()

    local function feval()
        return self.criterion.output, self.gradParams
    end
    --------------------------
    -- iterate over dataset --
    --------------------------

    local timer = torch.Timer()
    local losses = {}
    local confusion = optim.ConfusionMatrix(dataloader.classes[1])
    local dataSize = dataloader:size()
    for n, sample in dataloader:run(self.criterions, 0) do

        self:copyInputs(sample)
        local output = self.model:forward(self.input)
        local loss = self.criterion:forward(self.model.output, self.target)

        self.model:zeroGradParameters()
        self.criterion:backward(self.model.output, self.target)
        self.model:backward(self.input, self.criterion.gradInput)

        self.optim(feval, self.params, self.optimState)

        -- Log
        table.insert(losses, loss)
        for i = 1, self.target:size(1) do
            confusion:add(output[1][i], self.target[i][1])
        end
        if not clOpt.search then
            xlua.progress(n, dataSize)
        end
        if clOpt.debug == 1 then
            break
        end
    end

    -- Log
    local meanLoss = torch.Tensor(losses):mean()
    local speed = timer:time().real / dataloader.__size * 100
    confusion:__tostring__()
    print(string.format("TRAIN: total=%.3f%%, average=%.3f%%, Speed=%.3fs, LR=%s, Loss=%.4f",
                        confusion.totalValid * 100,
                        confusion.averageValid * 100,
                        speed,
                        self.optimState.learningRate,
                        meanLoss))
    if clOpt.log == 1 and clOpt.debug == 0 then
        print(confusion)
    end

    return meanLoss, confusion.totalValid * 100, speed
end

----------
-- Test --
----------

function Trainer:test(dataloader, clOpt)
    self.model:evaluate()
    local confusion = optim.ConfusionMatrix(dataloader.classes[1])

    local losses = {}
    local dataSize = dataloader:size()
    for n, sample in dataloader:run(self.criterions, 0) do
        self:copyInputs(sample)

        -- Size 1 breaks batchNorm
        if self.input:size()[1] == 1 then
            break
        end
        local output = self.model:forward(self.input)
        local loss = self.criterion:forward(self.model.output, self.target)

        -- Log
        losses[#losses + 1] = loss
        -- for i = 1, self.target:size(1) do
            -- confusion:add(output[1][i], self.target[i][1])
        -- end
        if not clOpt.search then
            xlua.progress(n, dataSize)
        end
        if clOpt.debug == 1 then
            break
        end
    end

    self.model:training()

    -- log
    confusion:__tostring__()
    print(string.format("TEST:  total=%.2f%%, average=%.2f%%", confusion.totalValid * 100, confusion.averageValid * 100))
    if clOpt.log == 1 and clOpt.debug == 0 then
        print(confusion)
    end
    return torch.Tensor(losses):mean(), confusion.totalValid * 100
end

----------------------
-- Extract features --
----------------------

function Trainer:confidence(dataloader)
    self.model:evaluate()
    local inputs, targets
    local dataSize = dataloader:size()
    for n, sample in dataloader:run(self.criterions, 1) do
        self:copyInputs(sample)
        local output = self.model:forward(self.input):float()
        local predictions = torch.Tensor(output:size(1))
        for i = 1, output:size(1) do
            predictions[i] = output[{i, self.target[i]}]
        end
        predictions = torch.exp(predictions)
        if inputs then
            inputs = torch.cat(inputs, predictions, 1)
            targets = torch.cat(targets, self.target, 1)
        else
            inputs = predictions
            targets = self.target:clone()
        end
        xlua.progress(n, dataSize)
    end
    return {inputs = torch.totable(inputs), targets = torch.totable(targets), paths = dataloader.dataset.paths}
end

function Trainer:features(dataloader)
    self.model:evaluate()
    local inputs, targets
    local dataSize = dataloader:size()
    for n, sample in dataloader:run(self.criterions, 1) do
        self:copyInputs(sample)
        local output = self.model:forward(self.input):float()
        if inputs then
            inputs = torch.cat(inputs, output, 1)
            targets = torch.cat(targets, self.target, 1)
        else
            inputs = output:clone()
            targets = self.target:clone()
        end
        xlua.progress(n, dataSize)
    end
    return {inputs = inputs, targets = targets, paths = dataloader.dataset.paths}
end

return M.Trainer
