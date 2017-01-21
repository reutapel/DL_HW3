require 'torch'
require 'gnuplot'
require 'cutorch'
require 'cunn'  
require 'rnn'

local nIters = 2000
local batchSize = 80
local rho = 10   		-- sequence_length  
local hiddenSize = 300
local lr = 0.0001
local nPredict = 200

--------------------------------------------------------
local x = torch.linspace(0,200, 2000)
local cosx = torch.cos(x)
gnuplot.pngfigure('cosx.png')
gnuplot.plot({'cos', cosx[{{1,200}}],'+-'})
gnuplot.plotflush()

local sequence = cosx:cuda()

--------------------------------------------------------
-- #offsets = batchSize. s.t: offsets[i] = element 0 of the sequence i
local offsets = {}
for i=1,batchSize do
   table.insert(offsets, math.ceil(math.random()* (sequence:size(1)-rho) )) -- choose random elements as x_0
end
offsets = torch.LongTensor(offsets)

-- print(offsets) io.read() -- behaves as breakpoint 
--------------------------------------------------------
rnn = nn.Sequential()
   :add(nn.Linear(1, hiddenSize))
   :add(nn.FastLSTM(hiddenSize, hiddenSize, rho))
   :add(nn.NormStabilizer())
   :add(nn.Linear(hiddenSize, 1))
   :add(nn.HardTanh())
rnn = nn.Sequencer(rnn)
rnn:training()
print(rnn)

rnn = rnn:cuda()
criterion = nn.MSECriterion():cuda()

--------------------------------------------------------
local gradOutputsZeroed = {}
for step=1,rho do
  gradOutputsZeroed[step] = torch.zeros(batchSize,1):cuda()
end
print(gradOutputsZeroed)

--------------------------------------------------------
local iteration = 1
while iteration < nIters do
   rnn:forget()
   local inputs, targets = {}, {}
   for step=1,rho do
      inputs[step] = sequence:index(1, offsets):view(batchSize,1) 	--batch of sequences
      offsets:add(1) -- go to the next element in the sequence
      for j=1,batchSize do
         if offsets[j] > sequence:size(1) then
            offsets[j] = 1
         end
      end
      targets[step] = sequence:index(1, offsets)
   end
   rnn:zeroGradParameters()  -- important
   local outputs = rnn:forward(inputs) -- forward all sequence
   local err = criterion:forward(outputs[rho], targets[rho])
   print(string.format("Iteration %d ; NLL err = %f ", iteration, err))
   local gradOutputs = criterion:backward(outputs[rho], targets[rho])
   gradOutputsZeroed[rho] = gradOutputs
   local gradInputs = rnn:backward(inputs, gradOutputsZeroed) -- since we interested in 1 prediction
   rnn:updateParameters(lr)
   iteration = iteration + 1
end

-------------------------------------------------------------------------
-- "sliding window" 
-- first sequence is 0 ... n . Than predict n + 1 
-- second 1 ... n +1 . Than predict n + 2
rnn:evaluate()
local predict = torch.FloatTensor(nPredict):cuda() -- array of the next cos values predictions
local predictOneStep = torch.FloatTensor(nPredict):cuda()
for step = 1, rho do
  predict[step] = sequence[step]
  predictOneStep[step] = sequence[step]
end

local start = torch.Tensor(rho,1,1):zero():cuda()
local startOneStep = torch.Tensor(rho,1,1):zero():cuda()
iteration = 0
while rho + iteration < nPredict do
  for step=1,rho do
    start[step] = predict:index(1,torch.LongTensor({step+iteration})):view(1,1)
	startOneStep[step] = sequence[step + iteration]
  end

  rnn:forget()
  
  local output = rnn:forward(start)
  predict[iteration+rho+1] = (output[rho]:float())[1][1]

  local outputOneStep = rnn:forward(startOneStep)
  predictOneStep[iteration + rho + 1] = (outputOneStep[rho]:float())[1][1]
  
  iteration = iteration + 1
end

gnuplot.pngfigure('predCos.png')
gnuplot.plot({'predictOneStep',predictOneStep,'+'},{'predict',predict,'+'},{'cos(x)',sequence:narrow(1,1,nPredict),'-'})
gnuplot.plotflush()

