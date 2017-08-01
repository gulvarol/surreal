require 'torch'
require 'cutorch'
require 'paths'
require 'xlua'
require 'optim'
require 'nn'
require 'nnx'
require 'nngraph'

paths.dofile('TrainPlotter.lua')
torch.setdefaulttensortype('torch.FloatTensor')

local opts = paths.dofile('opts.lua')
opt = opts.parse(arg)

local heatmapSize = opt.inSize[2]/4 --64

if(opt.upsample) then
    heatmapSize = 256
end

opt.data = paths.concat(opt.dataRoot, 'SURREAL/data', opt.datasetname)

local logDir = opt.logRoot .. '/' .. opt.datasetname
opt.save     = paths.concat(logDir, opt.dirName)
opt.cache    = paths.concat(logDir, 'cache')
opt.plotter  = TrainPlotter.new(paths.concat(opt.save, 'plot.json'))

os.execute('mkdir -p ' .. opt.save)
os.execute('mkdir -p ' .. opt.cache)

-- Dependent on the ground truth
if(opt.supervision == 'depth') then -- classification of depth bins
    opt.outSize = {heatmapSize, heatmapSize} -- 64 x 64
    opt.nOutChannels = opt.depthClasses + 1 -- 20
elseif(opt.supervision == 'segm') then
    opt.outSize = {heatmapSize, heatmapSize} -- 64 x 64
    opt.nOutChannels = opt.segmClasses -- 15
end

if(opt.training == 'scratch') then
    opt.netType = 'hg' -- hourglass
elseif(opt.training == 'pretrained') then
    print('Make sure you set your pre-trained network path correct with opt.retrain option.')
else
    error('Set opt.training to either scratch or pretrained.')
end

-- Continue stopped training
if(opt.continue) then --epochNumber has to be set for this option
    print('Continuing from epoch ' .. opt.epochNumber)
    opt.retrain = opt.save .. '/model_' .. opt.epochNumber -1 ..'.t7' -- overwrites opt.retrain
    opt.optimState = opt.save .. '/optimState_'.. opt.epochNumber -1  ..'.t7'
    local backupDir = opt.save .. '/delete' .. os.time()
    os.execute('mkdir -p ' .. backupDir)
    os.execute('cp ' .. opt.save .. '/train.log ' ..backupDir)
    os.execute('cp ' .. opt.save .. '/test.log ' ..backupDir)
    os.execute('cp ' .. opt.save .. '/plot.json ' ..backupDir)
end

print(opt)
torch.save(paths.concat(opt.save, 'opt' .. os.time() .. '.t7'), opt)
cutorch.setDevice(opt.GPU) -- by default, use GPU 1
torch.manualSeed(opt.manualSeed)
print('LR ' .. opt.LR)
print('Saving everything to: ' .. opt.save)

paths.dofile('util.lua')
paths.dofile('model.lua')
paths.dofile('data.lua')
paths.dofile('train.lua')
paths.dofile('test.lua')
paths.dofile('eval.lua')

if(opt.evaluate) then
    test()
else
    epoch = opt.epochNumber
    for i=1,opt.nEpochs do
        train()
        test()
        epoch = epoch + 1
    end
end
