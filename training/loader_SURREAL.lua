
local M = {}

local function getMatFile(path, str)
    return paths.dirname(path) .. '/' .. paths.basename(path, 'mp4') .. str .. '.mat'
end

-- NUMBER OF FRAMES --
local function getDuration(path) -- can get it with OpenCV instead
    local zrot
    if pcall(function() zrot = matio.load( getMatFile(path, '_info'), 'zrot') end) then
        return zrot:nElement()
    else
        if(opt.verbose) then print('Zrot not loaded ' .. path) end
        return 0
    end
end

local function loadCameraLoc(path)
    local camLoc
    if pcall(function() camLoc = matio.load( getMatFile(path, '_info'), 'camLoc') end) then
    else
        if(opt.verbose) then print('CamLoc not loaded ' .. path) end; return nil
    end
    if(camLoc == nil) then; return nil; end
    return camLoc[1]
end

-- RGB --
local function loadRGB(path, t)
    local cap = cv.VideoCapture{filename=path}
    cap:set{propId=1, value=t-1} --CV_CAP_PROP_POS_FRAMES
    local rgb 
    if pcall(function() _, rgb = cap:read{}; rgb = rgb:permute(3, 1, 2):float()/255; rgb = rgb:index(1, torch.LongTensor{3, 2, 1}) end) then
        return rgb
    else
        if (opt.verbose) then print('Img not opened ' .. path) end
        return nil
    end
end

-- JOINTS 2D --
local function loadJoints2D(path, t)
    local joints2D, vars
    if pcall(function() vars = matio.load( getMatFile(path, '_info'), 'joints2D') end) then
        -- [24 x 2] -- it was 0-based
        if pcall(function() joints2D = vars[{{}, {}, { t }}]:squeeze():t():add(1); joints2D = joints2D:index(1, torch.LongTensor(opt.jointsIx)) end) then -- Get joint indices we are interested in
        else print(path .. ' has weirdness (joints2D)' .. t); return nil end
        local zeroJoint2D = joints2D[{{}, {1}}]:eq(1):cmul(joints2D[{{}, {2}}]:eq(239)) -- Check if joints are all zeros.
        if zeroJoint2D:sum()/zeroJoint2D:nElement() == 1 then
            if(opt.verbose) then print('Skipping ' .. path .. '... (joints2D are all [0, 0])') end
            return nil
        end
    else
        if(opt.verbose) then print('Joints2D not loaded ' .. path) end
    end
    return joints2D
end

-- JOINTS 3D --
function loadJoints3D(path, t)
    local joints3D, vars
    if pcall(function() vars = matio.load( getMatFile(path, '_info'), 'joints3D') end) then
        if pcall(function() joints3D = vars[{{}, {}, { t }}]:squeeze():t(); joints3D = joints3D:index(1, torch.LongTensor(opt.jointsIx))  end) then       -- [24 x 3]
        else print(path .. ' has weirdness (joints3D)' .. t); return nil end
        local zeroJoint3D = joints3D[{{}, {1}}]:eq(0):cmul(joints3D[{{}, {2}}]:eq(0)):cmul(joints3D[{{}, {3}}]:eq(0)) -- Check if joints are all zeros.
        if zeroJoint3D:sum()/zeroJoint3D:nElement() == 1 then
            if(opt.verbose) then print('Skipping ' .. path .. '... (joints3D are all [0, 0])') end
            return nil
        end
    else
        if(opt.verbose) then print('Joints3D not loaded ' .. path) end
    end
    return joints3D
end

-- SEGMENTATION --
local function loadSegm(path, t)
    local segm
    if pcall(function() segm = matio.load( getMatFile(path, '_segm'), 'segm_' .. t) end) then -- [240 x 320]
    else
        if(opt.verbose) then print('Segm not loaded ' .. path) end;  return nil 
    end
    if(segm == nil) then; return nil; end
    segm = changeSegmIx(segm, {2, 12, 9, 2, 13, 10, 2, 14, 11, 2, 14, 11, 2, 2, 2, 1, 6, 3, 7, 4, 8, 5, 8, 5}) 
    return segm
end

-- DEPTH --
local function loadDepth(path, t, dPelvis)
    local depth, out, pelvis, mask, nForeground, lowB, upB
    if pcall(function() depth = matio.load( getMatFile(path, '_depth'), 'depth_' .. t) end) then -- [240 x 320]
    else
        if(opt.verbose) then print('Depth not loaded ' .. path) end;  return nil, nil
    end
    if(depth == nil) then; return nil, nil; end

    out = torch.zeros(depth:size())
    mask = torch.le(depth, 1e+3)  -- background =1.0000e+10
    nForeground = mask:view(-1):sum()  -- #foreground pixels
    lowB = -(opt.depthClasses - 1)/2
    upB = (opt.depthClasses - 1)/2

    local fgix = torch.le(depth, 1e3)
    local bgix = torch.gt(depth, 1e3)
    out[fgix] = torch.cmax(torch.cmin(torch.ceil(torch.mul(torch.add(depth[fgix], -dPelvis), 1/opt.stp)), upB), lowB) -- align and quantize
    out[bgix] = lowB-1 -- background class
    out = out:add(1+upB) -- so that it's between 0-19. It was [-10, -9, .. 0 .. 9].

    return out, nForeground 
end

M.getDuration   = getDuration
M.loadCameraLoc = loadCameraLoc
M.loadRGB       = loadRGB
M.loadJoints2D  = loadJoints2D
M.loadJoints3D  = loadJoints3D
M.loadSegm      = loadSegm
M.loadDepth     = loadDepth

return M
