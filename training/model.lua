require 'nn'
require 'cunn'
require 'optim'
require 'cudnn'

-- Criterion
criterion = nn.ParallelCriterion()
for st = 1, opt.nStack do -- 8
    criterion:add(cudnn.SpatialCrossEntropyCriterion())
end

-- Create Network
--    If preloading option is set, preload weights from existing models appropriately
--    If model has its own criterion, override.
if opt.retrain ~= 'none' then
    assert(paths.filep(opt.retrain), 'File not found: ' .. opt.retrain)
    print('Loading model from file: ' .. opt.retrain);
    model = loadDataParallel(opt.retrain, opt.nGPU) -- defined in util.lua
else
    paths.dofile('models/' .. opt.netType .. '.lua')
    print('=> Creating model from file: models/' .. opt.netType .. '.lua')
    model = createModel(opt.nGPU) -- for the model creation code, check the models/ folder
    if opt.backend == 'cudnn' then
        require 'cudnn'
        cudnn.convert(model, cudnn) --
    elseif opt.backend == 'cunn' then
        require 'cunn'
        model = model:cuda()
    elseif opt.backend ~= 'nn' then
        error'Unsupported backend'
    end
end

-- Learning to upsample
if(not opt.evaluate and opt.upsample) then -- not opt.continue
    local upsampling = nn.Sequential()
    upsampling:add(nn.SpatialUpSamplingBilinear({oheight=opt.heatmapSize, owidth=opt.heatmapSize}))
    upsampling:add(cudnn.ReLU(true))
    upsampling:add(cudnn.SpatialConvolution(opt.nOutChannels, opt.nOutChannels, 3, 3, 1, 1, 1, 1))
    cudnn.convert(upsampling, cudnn)
 
    cpModel = model
    model = nn.Sequential()
    model:add(cpModel)
    model:add(nn.FlattenTable()) 
    model:add(nn.ParallelTable()
         :add(nn.Identity())
         :add(nn.Identity())
         :add(nn.Identity())
         :add(nn.Identity())
         :add(nn.Identity())
         :add(nn.Identity())
         :add(nn.Identity())
         :add(upsampling)
         )
end

print('=> Model')
print(model)

print('=> Criterion')
print(criterion)

-- Convert model to CUDA
print('==> Converting model and criterion to CUDA')
model:cuda()
criterion:cuda()

cudnn.fastest = true
cudnn.benchmark = true

collectgarbage()
