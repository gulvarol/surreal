
local ffi = require 'ffi'
-------------------------------------------------------------------------------
-- Helpful functions for evaluation
-------------------------------------------------------------------------------

--if(opt.show) then
require 'image'
wz = 2 --zoom
local zm = wz*4
if(opt.upsample) then zm = wz end
--end


function tensorMax(input)
   local y = input:max()
   local i = input:float():eq(y):nonzero()
   return i[{{1}, {}}]
end

-- output: Bx15x64x64 probabilities for 15 classes, label: Bx64x64
function segmPerformance(output, labels)
    local nBatch = output:size(1)
    local nClasses = output:size(2)
    local iou = torch.Tensor(nClasses):zero()
    local pixelacc = torch.Tensor(nClasses):zero()
    local intersection = torch.Tensor(nClasses):zero()
    local union = torch.Tensor(nClasses):zero()
    local npositives = torch.Tensor(nClasses):zero()
    for b = 1, nBatch do
        local label = labels[b]
        dummy, pred = torch.max(output[b], 1)
        for  cl = 1, nClasses do
            local ix = torch.eq(label, cl)
            local npos = ix:sum()
            local tp = torch.eq(label, pred:float()):cmul(ix):sum()
            local p = torch.eq(pred, cl):sum()

            intersection[cl] = intersection[cl] + tp
            union[cl] = union[cl] + (npos + p - tp )
            npositives[cl] = npositives[cl] + npos
            if(npos + p - tp ~= 0) then
                iou[cl] = iou[cl] + tp / (npos + p - tp )
            end
            if(npos ~= 0) then
                pixelacc[cl] = pixelacc[cl] + (tp / npos)
            end
        end
    end
    return iou[{{2, 15}}]:mean()/nBatch, iou, pixelacc[{{2, 15}}]:mean()/nBatch, pixelacc, intersection, union, npositives
end

-- output: Bx20x64x64, label:Bx64x64
function depthRMSE(output, label)
    local nBatch = output:size(1)
    local nClasses = output:size(2)
    local rmse = 0
    for b = 1, nBatch do
        local ix = torch.ne(label[b], 1):expandAs(label[b]) -- foreground pixels
        local nForeground = ix:sum()
        local dummy, pred = torch.max(output[b], 1)
        if(label[b][ix]:size():size() ~= 0) then -- not empty
            rmse = rmse + torch.sqrt(torch.mean(torch.pow(label[b][ix]:unfold(1, 2, 2) - pred[ix]:unfold(1, 2, 2):float(), 2)))
        else
            print('?!')
            -- counter of not evaluated images
        end
    end
    return rmse/nBatch
end

function evalPerf(inputsCPU, labelsCPU, outputs)
    local iouBatch = 0
    local iouBatchParts = torch.Tensor(opt.segmClasses):zero()
    local pxlaccBatch = 0
    local pxlaccBatchParts = torch.Tensor(opt.segmClasses):zero()
    local intersectionBatch = torch.Tensor(opt.segmClasses):zero()
    local unionBatch = torch.Tensor(opt.segmClasses):zero()
    local npositives = torch.Tensor(opt.segmClasses):zero()
    local rmse = 0

    if(opt.nStack > 0) then
        assert(#outputs == opt.nStack) -- each table entry is another stack 
        outputs = outputs[opt.nStack] -- take the last stack output
    end

    scoresCPU = outputs:float()

    if(opt.supervision == 'segm') then
        scoresSegm = scoresCPU
        labelsSegm = labelsCPU
        iouBatch, iouBatchParts, pxlaccBatch, pxlaccBatchParts, intersectionBatch, unionBatch, npositives = segmPerformance(scoresSegm, labelsSegm)
        if(opt.show) then
            require 'image'
            for i = 1, opt.batchSize do
                dummy, pred = torch.max(scoresSegm[i], 1) 
                im = pred:float() -- 1 x 64 x 64
                im[1][1][1] = 1
                im[1][1][2] = opt.segmClasses
                wOutputSegm = image.display({image=image.y2jet(im), win=wOutputSegm, zoom=zm, legend='PRED SEGM'})
            end
        end
    end

    if(opt.supervision == 'depth') then
        scoresDepth = scoresCPU
        labelsDepth = labelsCPU
        rmse = depthRMSE(scoresDepth, labelsDepth)
        if(opt.show) then
            require 'image'
            for i = 1, opt.batchSize do
                dummy, pred = torch.max(scoresDepth[i], 1)
                im = pred:float() -- 1 x 64 x 64
                im[1][1][1] = 1
                im[1][1][2] = opt.depthClasses + 1
                wOutputDepth = image.display({image=image.y2jet(im), win=wOutputDepth, zoom=zm, legend='PRED DEPTH'})
            end
        end
    end

    if opt.show then
        sys.sleep(2)
    end
  
    return pxlaccBatch, iouBatch, intersectionBatch, unionBatch, npositives, rmse
end