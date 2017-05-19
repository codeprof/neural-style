require 'torch'
require 'nn'
require 'cutorch'
require 'cunn'
      



function GramMatrix()
  local net = nn.Sequential()

  net:add(nn.View(-1):setNumInputDims(2))
  local concat = nn.ConcatTable()
  concat:add(nn.Identity())
  concat:add(nn.Identity())
  net:add(concat)
  net:add(nn.MM(false, true)) 
  return net:cuda()
end


torch.manualSeed(1)


for i=0,4 do

	x =  torch.load('content'..i, 'ascii'):cuda()
	Gy = torch.load('style'..i, 'ascii'):cuda()

	print(1)
	gram = GramMatrix():cuda()
	print(1)
	crit = nn.MSECriterion():cuda()
	print(1)

	for t=1,2000 do
		Gx = gram:forward(x)
		Gx:div(x:nElement())
		loss = crit:forward(Gx, Gy)
		print(t .. ' ' .. loss)
		
		local d = crit:backward(Gx, Gy)

		grad = gram:backward(x, d)

		x = (x-grad*0.1)
	end
	torch.save('merge'..i,x:double(), 'ascii')
end	
