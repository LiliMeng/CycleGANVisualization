require 'torch'
require 'nn'
require 'lfs'
require 'image'
require 'loadcaffe'
require 'InstanceNormalization'
require 'cutorch'
utils = require 'misc.utils'

cutorch.setDevice(1)
cmd = torch.CmdLine()
cmd:text('Options')


function preprocess(img)
    -- RGB to BGR
    if img:size(1) == 3 then
      local perm = torch.LongTensor{3, 2, 1}
      img = img:index(1, perm)
    end
    -- [0,1] to [-1,1]
    img = img:mul(2):add(-1)

    -- check that input is in expected range
    assert(img:max()<=1,"badly scaled inputs")
    assert(img:min()>=-1,"badly scaled inputs")

    return img
end

function scaleImage(input, loadSize)

    -- replicate bw images to 3 channels
    if input:size(1)==1 then
      input = torch.repeatTensor(input,3,1,1)
    end

    input = image.scale(input, loadSize, loadSize)

    return input
end

function loadImage(path, loadSize, nc)
  local input = image.load(path, 3, 'float')
  input= preprocess(scaleImage(input, loadSize))

  if nc == 1 then
    input = input[{{1}, {}, {}}]
  end

  return input
end

-- Model parameters 
cmd:option('-model_file', 'models/latest_net_D_A.t7')
cmd:option('-backend', 'nn')

cmd:option('-input_image_path', 'images/0012.jpg', 'Input image path')
cmd:option('-output_image_name', '', 'Output image name')

-- Miscellaneous
cmd:option('-seed', 123, 'Torch manual random number generator seed')
cmd:option('-gpuid', 0, '0-indexed id of GPU to use. -1 = CPU')
cmd:option('-out_path', 'output/', 'Output path')

-- Parse command-line parameters
opt = cmd:parse(arg or {})
print(opt)

torch.manualSeed(opt.seed)
torch.setdefaulttensortype('torch.FloatTensor')
lfs.mkdir(opt.out_path)

if opt.gpuid >= 0 then
  require 'cunn'
  require 'cutorch'
  cutorch.setDevice(opt.gpuid + 1)
  cutorch.manualSeed(opt.seed)
end

-- Load image
--local img = utils.preprocess(opt.input_image_path, 128, 128)

img = loadImage(opt.input_image_path,128,1)
img = torch.reshape(img, torch.LongStorage{1, 1, 128, 128})


print("no problem of loading images")

-- Load CNN

path = "./models/latest_net_D_A.t7"
cnn = torch.load(path)
print("the cnn architecture is:")
print(cnn)

-- remove one layer is to visualize the layer before last layer
--cnn:remove()

print("after layer removal,the cnn architecture is:")
print(cnn)


-- Clone & replace ReLUs for Guided Backprop
local cnn_gb = cnn:clone()
cnn_gb:replace(utils.guidedbackprop)


-- Transfer to GPU
if opt.gpuid >= 0 then
  cnn:cuda()
  cnn_gb:cuda()
  img = img:cuda()
else
  img = img:float()
end

print("no problem before forward pass output")

-- Forward pass
local output = cnn:forward(img)

local output_gb = cnn_gb:forward(img)
print("no problem after forward pass output")



print("cnn.modules[#cnn.modules]")
print(cnn.modules[#cnn.modules])
print("cnn.modules[#cnn.modules].output:size()")
print(cnn.modules[#cnn.modules].output:size())
-- Synthesize grad output
doutput=cnn.modules[#cnn.modules].output:clone()
doutput:fill(0.1)


-- Guided Backprop
local gb_viz = cnn_gb:backward(img, doutput)
print("guided backprop is done")
print(gb_viz:size())

-- Save the guided image 
image.save(opt.out_path .. 'classify_gb_' .. opt.output_image_name .. '.png', image.toDisplayTensor(gb_viz))

