-- Setup a reused optimization state. If needed, reload from disk
if not optimState then
    optimState = {
        learningRate = opt.LR,
        learningRateDecay = 0.0,
        momentum = opt.momentum,
        weightDecay = opt.weightDecay,
        dampening = 0.0,
        alpha = opt.alpha,
        epsilon = opt.epsilon
    }
end

if opt.optimState ~= 'none' then
    assert(paths.filep(opt.optimState), 'File not found: ' .. opt.optimState)
    print('Loading optimState from file: ' .. opt.optimState)
    optimState = torch.load(opt.optimState)
    optimState.learningRate = opt.LR -- update LR
end

-- Logger
trainLogger = optim.Logger(paths.concat(opt.save, 'train.log'))
local batchNumber
local loss, pxlacc, iou, nanCntPxl, nanCntIou

-- train - this function handles the high-level training loop,
--            i.e. load data, train model, save model and state to disk
function train()
    print('==> doing epoch on training data:')
    print("==> online epoch # " .. epoch)
    trainEpochLogger = optim.Logger(paths.concat(opt.save, ("epoch_%d_train.log"):format(epoch)))

    batchNumber = 0
    cutorch.synchronize()

    -- set to training mode
    model:training()
    model:cuda()

    local tm = torch.Timer()
    loss         = 0
    rmse         = 0
    intersection = torch.Tensor(opt.segmClasses):zero()
    union        = torch.Tensor(opt.segmClasses):zero()
    npos         = torch.Tensor(opt.segmClasses):zero()
    pxlacc       = torch.Tensor(opt.segmClasses):zero()
    iou          = torch.Tensor(opt.segmClasses):zero()
    for i=1,opt.epochSize do
        -- queue jobs to data-workers
        donkeys:addjob(
            -- the job callback (runs in data-worker thread)
            function()
                local inputs, labels, indices = trainLoader:sample(opt.batchSize)
                return inputs, labels, indices
            end,
            -- the end callback (runs in the main thread)
            trainBatch
        )
    end

    donkeys:synchronize()
    cutorch.synchronize()

    -- Performance measures
    loss = loss / opt.epochSize
    iou = 100*torch.cdiv(intersection, union)
    pxlacc = 100*torch.cdiv(intersection, npos)
    m_iou = iou[{{2, opt.segmClasses}}]:mean()
    m_pxlacc = pxlacc[{{2, opt.segmClasses}}]:mean()
    rmse = rmse / opt.epochSize

    trainLogger:add{
        ['epoch'] = epoch,
        ['loss'] = loss,
        ['LR'] = optimState.learningRate,
        ['pxlacc'] = table2str(pxlacc),
        ['iou'] = table2str(iou),
        ['m_iou'] = m_iou,
        ['m_pxlacc'] = m_pxlacc,
        ['rmse'] = rmse
    }
    opt.plotter:add('LR', 'train', epoch, optimState.learningRate)
    opt.plotter:add('loss', 'train', epoch, loss)
    opt.plotter:add('pxlacc', 'train', epoch, m_pxlacc)
    opt.plotter:add('iou', 'train', epoch, m_iou)
    opt.plotter:add('rmse', 'train', epoch, rmse)

    print(string.format('Epoch: [%d][TRAINING SUMMARY] Total Time(s): %.2f\t'
        .. 'Loss: %.6f \t'
        .. 'IOU: %.2f \t'
        .. 'PixelAcc: %.2f \t'
        .. 'RMSE: %.2f \t',
        epoch, tm:time().real, loss, m_iou, m_pxlacc, rmse))
    print('\n')
    collectgarbage()
    model:clearState()
    saveDataParallel(paths.concat(opt.save, 'model_' .. epoch .. '.t7'), model)
    torch.save(paths.concat(opt.save, 'optimState_' .. epoch .. '.t7'), optimState)
end -- of train()
-------------------------------------------------------------------------------------------

-- GPU inputs (preallocate)
local inputs = torch.CudaTensor()
local labels = torch.CudaTensor()

local timer = torch.Timer()
local dataTimer = torch.Timer()

local parameters, gradParameters = model:getParameters()

-- Used by train() to train a single batch after the data is loaded.
function trainBatch(inputsCPU, labelsCPU, instancesCPU, indicesCPU)
    cutorch.synchronize()
    collectgarbage()
    local dataLoadingTime = dataTimer:time().real
    timer:reset()
     
    -- transfer over to GPU
    inputs:resize(inputsCPU:size()):copy(inputsCPU)
    labels:resize(labelsCPU:size()):copy(labelsCPU)

    local err, outputs, target

    feval = function(x)
        model:zeroGradParameters()
        outputs = model:forward(inputs) -- table of nStack outputs
        if(opt.nStack > 1) then
            target = {}
            if (opt.upsample) then
                require 'image'
                local lowSize = {math.ceil(opt.inSize[3]/4), math.ceil(opt.inSize[2]/4)}
                local labelSmall = torch.zeros(labels:size(1), labels:size(2), lowSize[2], lowSize[1])
                for b = 1, labels:size(1) do -- batchSize
                    labelSmall[b]:copy(image.scale(labels[b]:float(), lowSize[1], lowSize[2], 'simple'))
                end
                -- First 7 stacks, 4* low resolution
                for st = 1, opt.nStack-1 do
                    table.insert(target, labelSmall:cuda())
                end
                -- 8th stack high resolution
                table.insert(target, labels)
            else
                -- Same ground truth for all 8 stacks
                for st = 1, opt.nStack do
                    table.insert(target, labels)
                end
            end
        else
            -- No stack
            target = labels
        end

        err = criterion:forward(outputs, target) 
        local gradOutputs = criterion:backward(outputs, target)

        model:backward(inputs, gradOutputs)
        return err, gradParameters
    end

    optim.rmsprop(feval, parameters, optimState)

    local pxlaccBatch, iouBatch, intersectionBatch, unionBatch, nposBatch, rmseBatch = evalPerf(inputsCPU, labelsCPU, outputs, instancesCPU)
    
    local str = string.format("PixelAcc: %.2f,\tIOU: %.2f,\tRMSE: %.2f", 100*pxlaccBatch, 100*iouBatch, rmseBatch)

    if (intersectionBatch == intersectionBatch) then intersection = torch.add(intersection, intersectionBatch) end
    if (unionBatch == unionBatch)               then union        = torch.add(union, unionBatch) end
    if (nposBatch == nposBatch)                 then npos         = torch.add(npos, nposBatch)  end
    if (rmseBatch == rmseBatch)                 then rmse         = rmse + rmseBatch  end

    cutorch.synchronize()
    batchNumber = batchNumber + 1
    loss = loss + err
   
    print(('Epoch: [%d][%d/%d] Time: %.3f, Err: %.2f \t %s, \t LR: %.0e, \t DataLoadingTime %.3f'):format(
        epoch, batchNumber, opt.epochSize, timer:time().real, err, str,
        optimState.learningRate, dataLoadingTime))

    trainEpochLogger:add{
        ['BatchNumber'] = string.format("%d", batchNumber),
        ['Error'] = string.format("%.8f", err),
        ['LR'] = string.format("%.0e", optimState.learningRate)
    }
    dataTimer:reset()
end
