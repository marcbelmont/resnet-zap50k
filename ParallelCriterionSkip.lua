local ParallelCriterionSkip, parent = torch.class('nn.ParallelCriterionSkip', 'nn.Criterion')

function ParallelCriterionSkip:__init(repeatTarget)
    parent.__init(self)
    self.criterions = {}
    self.weights = {}
    self.gradInput = {}
    self.repeatTarget = repeatTarget
end

function ParallelCriterionSkip:add(criterion, weight)
    assert(criterion, 'no criterion provided')
    weight = weight or 1
    table.insert(self.criterions, criterion)
    table.insert(self.weights, weight)
    return self
end

function ParallelCriterionSkip:updateOutput(input, target)
    self.output = 0
    target = target:t(1, 2)
    for i, criterion in ipairs(self.criterions) do
        local target = self.repeatTarget and target or target[i]
        -- skip
        if target:min() == -1 then
            if target:max() > 0 then
                -- Update output only if the target there's a target (>0)
                local indices =  torch.range(1, target:size()[1]):long()[target:gt(0)]
                local target = target:index(1, indices)
                local input = input[i]:index(1, indices)
                self.output = self.output + self.weights[i] * criterion:updateOutput(input, target)
            end
        else
            self.output = self.output + self.weights[i] * criterion:updateOutput(input[i], target)
        end
    end
    return self.output
end

function ParallelCriterionSkip:updateGradInput(input, target)
    self.gradInput = nn.utils.recursiveResizeAs(self.gradInput, input)
    nn.utils.recursiveFill(self.gradInput, 0)
    target = target:t(1, 2)
    for i, criterion in ipairs(self.criterions) do
        local target = self.repeatTarget and target or target[i]
        -- skip
        if target:min() == -1 then
            if target:max() > 0 then
                -- Mask gradInput if there's no label (-1)
                local indices =  torch.range(1, target:size()[1]):long()[target:eq(-1)]
                target:indexFill(1, indices, 1)
                local t2 = criterion:updateGradInput(input[i], target)
                t2:indexFill(1, indices, 0)
                nn.utils.recursiveAdd(self.gradInput[i], self.weights[i], t2)
            end
        else
            nn.utils.recursiveAdd(self.gradInput[i], self.weights[i], criterion:updateGradInput(input[i], target))
        end
    end
    return self.gradInput
end

function ParallelCriterionSkip:type(type, tensorCache)
    self.gradInput = {}
    return parent.type(self, type, tensorCache)
end
