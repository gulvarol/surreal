require 'image'
require 'cudnn'
require 'nngraph'
require 'cunn'
local matio = require 'matio'
torch.setdefaulttensortype('torch.FloatTensor')
local meanStd   = torch.load('../training/meanstd/meanRgb.t7')

-- Load pre-trained models
local modelSegm = torch.load('../training/models/model_segm_png.t7')
local modelDepth = torch.load('../training/models/model_depth_png.t7')

-- Put models in evaluate mode
modelSegm:evaluate()
modelDepth:evaluate()

-- Load the input image
local imgPath = 'samples/input1.png'
local rgb = image.load(imgPath)
local h = rgb:size(2)
local w = rgb:size(3)

-- Resize and center crop to have 256x256 image
-- Here it helps to center the fully visible human body
local inputSc, inputCropped
if(h > w) then
	local newH = math.ceil(256*h/w)
	local st = math.ceil((newH - 256)/2)+1
	inputSc = image.scale(rgb, 256, newH)
	inputCropped = inputSc[{{}, {st, st+255}, {}}]:clone()
else
	local newW = math.ceil(256*w/h)
	local st = math.ceil((newW - 256)/2)+1
	inputSc = image.scale(rgb, newW, 256)
	inputCropped = inputSc[{{}, {}, {st, st+255}}]:clone()
end
local input = image.scale(inputCropped, 256, 256):view(1, 3, 256, 256)

-- Transfer the tensor to GPU
local inputGPU = input:cuda()

-- Whitening
for j = 1, #meanStd.mean do -- for each channel
    inputGPU[{{}, {j}, {}, {}}]:add(-meanStd.mean[j])
    inputGPU[{{}, {j}, {}, {}}]:div(meanStd.std[j])
end

-- Apply the models
local lowSegm = modelSegm:forward(inputGPU)
local lowDepth = modelDepth:forward(inputGPU)

-- Upsample the heatmap outputs from the 8th (last) stack
local highSegm = image.scale(lowSegm[8][1]:float(), 256, 256)
local highDepth = image.scale(lowDepth[8][1]:float(), 256, 256)

-- Max over heatmap channels
local dummy, predSegm = torch.max(highSegm, 1)
local dummy, predDepth = torch.max(highDepth, 1)

--predSegm[1][1][1]=1
--predSegm[1][1][2]=15
--predDepth[1][1][1]=1
--predDepth[1][1][2]=20

-- Visualize input/output
image.display({image=image.y2jet(predSegm:float()), legend='Segm output'})
image.display({image=image.y2jet(predDepth:float()),legend='Depth output'})
image.display({image=input, legend='RGB input'})

-- Save predictions
matio.save('samples/segm1_pred.mat', predSegm)
matio.save('samples/depth1_pred.mat', predDepth)
