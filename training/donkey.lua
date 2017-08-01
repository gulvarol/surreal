paths.dofile('dataset.lua')
paths.dofile('util.lua')
paths.dofile('img.lua')
paths.dofile('eval.lua')

local loadSize = opt.loadSize
local inSize   = opt.inSize

require 'image'
matio = require 'matio'
cv = require 'cv'
require 'cv.videoio'

local zm
if opt.show then
    require 'qtwidget'
    wz = 2 -- window zoom
    zm = wz*4
    if(opt.upsample) then zm = wz end
    wr = qtwidget.newwindow(inSize[3]*wz, inSize[2]*wz, 'RGB')
    if opt.supervision == 'depth' then
        wd = qtwidget.newwindow(inSize[3]*wz, inSize[2]*wz, 'DEPTH')
    end
    if opt.supervision == 'segm' then
        ws = qtwidget.newwindow(inSize[3]*wz, inSize[2]*wz, 'SEGM')
    end
end

-- This file contains the data-loading logic and details.
-- It is run by each data-loader thread.
------------------------------------------

-- a cache file of the training metadata (if doesnt exist, will be created)
local trainCache = paths.concat(opt.cache, opt.trainDir .. 'Cache.t7')
local testCache = paths.concat(opt.cache, opt.testDir .. 'Cache.t7')
local meanstdCache = paths.concat(opt.cache, 'meanstdCache.t7')

-- Check for existence of opt.data
if not os.execute('cd ' .. opt.data) then
    error(("could not chdir to '%s'"):format(opt.data))
end

-- Merge body parts
function changeSegmIx(segm, s)
    local out = torch.zeros(segm:size())
    for i = 1,#s do out[segm:eq(i)] = s[i] end
    return out
end

-- Loading functions
local loader = paths.dofile('loader_SURREAL.lua')

-- Mean/std for whitening
local meanstd = torch.load(paths.thisfile('meanstd/meanRgb.t7'))
local mean = meanstd.mean
local std = meanstd.std

-- function common to do processing on loaded image/label-----------------------------------
local Hook = function(self, path, set)
    collectgarbage()
    local rgbFull, rgb, label
    local joints2D, joints3D, pelvis3D
    local depth, depthFull, nForeground, camLoc
    local segm, segmFull

    local t1
    local iT = loader.getDuration(path)
    if(paths.extname(path) == 'jpg') then
        local matPath
        t1, matPath = loader.parseRGBPath(path)
    else
        if(set == 'train') then -- take random
            t1 = math.ceil(torch.uniform(1, iT))
        elseif(set == 'test') then -- take middle
            t1 = math.ceil(iT/2)
        end
    end

    -- load input
    rgbFull = loader.loadRGB(path, t1)

    -- load 2D joints to determine the bounding box
    joints2D = loader.loadJoints2D(path, t1) -- [ 2 x nJoints]
    
    -- depthFull and depth are quantized depths before and after cropping
    if(opt.supervision == 'depth') then
        local dPelvis -- depth of the pelvis to align depth image
        local joints3D = loader.loadJoints3D(path, t1) -- [ 3 x nJoints] 
        if(joints3D ~= nil) then
            pelvis3D = joints3D[7]:clone()
        end
        camLoc = loader.loadCameraLoc(path) -- [3]
        if(camLoc ~= nil and pelvis3D ~= nil) then
            dPelvis = camLoc[1] - pelvis3D[1] --- camLoc(x) - pelvis3D(x) 
        end
        
        if(dPelvis ~= nil) then
            depthFull, nForeground = loader.loadDepth(path, t1, dPelvis)
        end
    end
 
    -- segmFull and segm are segmentation masks before and after cropping
    if(opt.supervision == 'segm') then
        segmFull = loader.loadSegm(path, t1)
    end

    -- Check
    if(rgbFull == nil or joints2D == nil or 
        (opt.supervision == 'depth' and depthFull == nil) or 
        (opt.supervision == 'segm' and segmFull == nil)) then
            if(opt.verbose) then print('Nil! ' .. path) end
            return nil, nil
    end

    -- Crop, scale
    local rot = 0
    local scale = getScale(joints2D, rgbFull:size(2) )
    local center = getCenter(joints2D)
    if (center[1] < 1 or center[2] < 1 or center[1] > loadSize[3] or center[2] > loadSize[2]) then
        if(opt.verbose) then print('Human out of image ' .. path .. ' center: ' .. center[1] .. ', ' .. center[2]) end
        return nil, nil
    end

    -- Scale and rotation augmentation (randomly samples on a normal distribution)
    if(set == 'train') then
        local function rnd(x) return math.max(-2*x,math.min(2*x,torch.randn(1)[1]*x)) end
        scale = scale * (2 ^ rnd(opt.scale))
        rot = rnd(opt.rotate)
        if torch.uniform() <= .6 then rot = 0 end
    end

    -- Crop
    rgb = crop(rgbFull, center, scale, rot, inSize[2], 'bilinear') -- square

    -- Color augmentation
    if(set == 'train') then
        for c=1, 3 do
            rgb[{{c}, {}, {}}]:mul(torch.uniform(0.6,1.4)):clamp(0,1)
        end
    end

    if opt.show then
        wr = image.display({image=rgb, win=wr, zoom=wz})
        sys.sleep(0.1)
    end

    for c = 1, #mean do
        if mean then rgb[{{c}, {}, {}}]:add(-mean[c]) end
        if  std then rgb[{{c}, {}, {}}]:div(std[c]) end
    end

    if(opt.supervision == 'depth') then
        depth = crop(depthFull, center, scale, rot, opt.outSize[1], 'simple')
        depth = depth + 1
        -- it was kept [0-19] until cropping because it puts zero when rotating.
        -- we don't want 0 because they are class indices (should be positive)
        label = depth
        if opt.show then
            depth[{{1}, {1}}] = 1
            depth[{{1}, {2}}] = opt.depthClasses + 1
            wd = image.display({image=image.y2jet(depth), win=wd, zoom=zm})
        end
    end

    if(opt.supervision == 'segm') then
        segm = crop(segmFull, center, scale, rot, opt.outSize[1], 'simple')
        segm = segm + 1 -- same trick as in depth
        label = segm
        if opt.show then
          ws = image.display({image=image.y2jet(segm), win=ws, zoom=zm})
        end
    end

    collectgarbage()
    return rgb, label
end

--trainLoader & function to load the train image-----------------------------------
trainHook = function(self, path)
    return Hook(self, path, 'train')
end

if paths.filep(trainCache) then
    print('Loading train metadata from cache')
    trainLoader = torch.load(trainCache)
    trainLoader.sampleHookTrain = trainHook
    --assert(trainLoader.paths[1] == paths.concat(opt.data, 'train'),
    --       'cached files dont have the same path as opt.data. Remove your cached files at: '
    --          .. trainCache .. ' and rerun the program')
else
    print('Creating train metadata')
    trainLoader = dataLoader{
        paths = {paths.concat(opt.data, opt.trainDir)},
        split = 100,
        verbose = true,
        forceClasses = opt.forceClasses
    }
    torch.save(trainCache, trainLoader)
    trainLoader.sampleHookTrain = trainHook
end
collectgarbage()

-- do some sanity checks on trainLoader
do
    local class = trainLoader.imageClass
    local nClasses = #trainLoader.classes
    assert(class:max() <= nClasses, "class logic has error")
    assert(class:min() >= 1, "class logic has error")
end

--testLoader & function to load the test image-----------------------------------
testHook = function(self, path)
    return Hook(self, path, 'test')
end

if paths.filep(testCache) then
    print('Loading test metadata from cache')
    testLoader = torch.load(testCache)
    testLoader.sampleHookTest = testHook
    assert(testLoader.paths[1] == paths.concat(opt.data, opt.testDir),
        'cached files dont have the same path as opt.data. Remove your cached files at: '
        .. testCache .. ' and rerun the program')
else
    print('Creating test metadata')
    print('Test dir: ' .. opt.testDir)
    testLoader = dataLoader{
        paths = {paths.concat(opt.data, opt.testDir)},
        split = 0,
        verbose = true,
        forceClasses = trainLoader.classes
    }
    torch.save(testCache, testLoader)
    testLoader.sampleHookTest = testHook
end
collectgarbage()
--------------------------------------------------------------------------------
