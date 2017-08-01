if(opt.evaluate) then
    testLogger = optim.Logger(paths.concat(opt.save, 'evaluate.log'))
else
    testLogger = optim.Logger(paths.concat(opt.save, opt.testDir .. '.log'))
end

local batchNumber
local loss, pxlacc, iou, nanCntPxl, nanCntIou
local timer = torch.Timer()

function test()
    local optimState 
    if(opt.evaluate) then
        print('==> Testing final predictions')
        if(opt.saveScores) then scoreFile = io.open(paths.concat(opt.save, 'outputs.log'), "w") end
    else
        optimState = torch.load(paths.concat(opt.save, 'optimState_' .. epoch .. '.t7'))
        print('==> validation epoch # ' .. epoch)
    end

    batchNumber = 0
    cutorch.synchronize()
    timer:reset()

    -- set to evaluate mode
    model:evaluate()

    loss         = 0
    rmse         = 0
    intersection = torch.Tensor(opt.segmClasses):zero()
    union        = torch.Tensor(opt.segmClasses):zero()
    npos         = torch.Tensor(opt.segmClasses):zero()
    pxlacc       = torch.Tensor(opt.segmClasses):zero()
    iou          = torch.Tensor(opt.segmClasses):zero()
    for i=1,nTest/opt.batchSize do -- nTest is set in data.lua
        local indexStart = (i-1) * opt.batchSize + 1
        local indexEnd = math.min(nTest, indexStart + opt.batchSize - 1)
        donkeys:addjob(
            -- work to be done by donkey thread
            function()
                local inputs, labels, indices = testLoader:get(indexStart, indexEnd)
                return inputs, labels, indices
            end,
            -- callback that is run in the main thread once the work is done
            testBatch
        )
    end

    donkeys:synchronize()
    cutorch.synchronize()

    -- Performance measures:
    loss = loss / (nTest/opt.batchSize)
    iou = 100*torch.cdiv(intersection, union)
    pxlacc = 100*torch.cdiv(intersection, npos)
    m_iou = iou[{{2, opt.segmClasses}}]:mean()
    m_pxlacc = pxlacc[{{2, opt.segmClasses}}]:mean()
    rmse = rmse / (nTest/opt.batchSize)

    testLogger:add{
        ['epoch'] = epoch,
        ['loss'] = loss,
        ['pxlacc'] = table2str(pxlacc),
        ['iou'] = table2str(iou),
        ['m_iou'] = m_iou,
        ['m_pxlacc'] = m_pxlacc,
        ['rmse'] = rmse
    }
    if(not opt.evaluate) then
        opt.plotter:add('loss', 'test', epoch, loss)
        opt.plotter:add('pxlacc', 'test', epoch, m_pxlacc)
        opt.plotter:add('iou', 'test', epoch, m_iou)
        opt.plotter:add('rmse', 'test', epoch, rmse)
        print(string.format('Epoch: [%d] ', epoch))
    elseif(opt.saveScores) then scoreFile:close() 
    end

    print(string.format('[TESTING SUMMARY] Total Time(s): %.2f \t'
        .. 'Loss: %.6f\t'
        .. 'IOU: %.2f\t'
        .. 'PixAcc: %.2f\t'
        .. 'RMSE: %.2f\t',
        timer:time().real, loss, m_iou, m_pxlacc, rmse))
    print('\n')

end -- of test()
-----------------------------------------------------------------------------

local inputs = torch.CudaTensor()
local labels = torch.CudaTensor()

function testBatch(inputsCPU, labelsCPU, instancesCPU, indicesCPU)
    batchNumber = batchNumber + opt.batchSize

    inputs:resize(inputsCPU:size()):copy(inputsCPU)
    labels:resize(labelsCPU:size()):copy(labelsCPU)

    local outputs = model:forward(inputs)

    local target
    if(opt.nStack > 1) then
        target = {}
        if (opt.upsample) then
            require 'image'
            local lowSize = {math.ceil(opt.inSize[3]/4), math.ceil(opt.inSize[2]/4)}
            local labelSmall = torch.zeros(labels:size(1), labels:size(2), lowSize[2], lowSize[1])
            for b = 1, labels:size(1) do -- in batch
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
        target = labels
    end

    -- Compute loss
    local err = criterion:forward(outputs, target) 

    local pxlaccBatch, iouBatch, intersectionBatch, unionBatch, nposBatch, rmseBatch = evalPerf(inputsCPU, labelsCPU, outputs, instancesCPU)

    local str = string.format("PixelAcc: %.2f,\tIOU: %.2f,\tRMSE: %.2f", 100*pxlaccBatch, 100*iouBatch, rmseBatch)
    if (intersectionBatch == intersectionBatch) then intersection = torch.add(intersection, intersectionBatch) end
    if (unionBatch == unionBatch)               then union        = torch.add(union, unionBatch) end
    if (nposBatch == nposBatch)                 then npos         = torch.add(npos, nposBatch)  end
    if (rmseBatch == rmseBatch)                 then rmse         = rmse + rmseBatch  end

    if(opt.saveScores and opt.evaluate) then
        scoresCPU = scoresCPU:view(opt.batchSize, -1)
        labelsCPU = labelsCPU:view(opt.batchSize, -1)
        for i=1,scoresCPU:size(1) do
            scoreFile:write(string.format('%d', indicesCPU[i]))
            for jj=1,labelsCPU:size(2) do -- 101
                scoreFile:write('\t' .. string.format('%11.4e', labelsCPU[i][jj]))
            end
            for jj=1,scoresCPU:size(2) do -- 101
                scoreFile:write('\t' .. string.format('%11.4e', scoresCPU[i][jj]))
            end
            scoreFile:write('\n')
            scoreFile:flush()
        end
    end

    cutorch.synchronize()
    loss = loss + err

    if(opt.evaluate) then
        print(string.format('Testing [%d/%d] \t Loss %.8f \t %s', batchNumber, nTest, err, str))
    else
        print(string.format('Epoch: Testing [%d][%d/%d] \t Loss %.8f \t %s', epoch, batchNumber, nTest, err, str))
    end
end
