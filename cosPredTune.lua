require 'torch'
require 'gnuplot'
require 'cutorch'
require 'cunn'  
require 'rnn'

local nb_layers = 2
local hidden_size = 300
local nIndex = 1
local rho = 10
local dropout = 0.5
local nPredict = 200
local eval_one_step = 'false'
local iters = 20000
local batch_size = 80
local learning_rate = 0.0001
local model = 'lstm'

function buildModel()
  local rnn

  if model == 'lstm' then
	  rnn = nn.Sequential()
		 :add(nn.FastLSTM(nIndex, hidden_size), rho)
		 :add(nn.NormStabilizer())

	  for i = 2, nb_layers do
		 rnn:add(nn.FastLSTM(hidden_size, hidden_size), rho)
		 :add(nn.NormStabilizer())
	  end
  end

  if model == 'grid-lstm' then -- this requires pull request.
      rnn = nn.Sequential()
        :add(nn.Linear(nIndex, hidden_size))
        :add(nn.Grid2DLSTM(hidden_size, nb_layers, dropout, true, rho))
  end
    

  rnn = nn.Sequential()
	  :add(nn.SplitTable(0,2))
	  :add(nn.Sequencer(rnn))   -- add the rnn
	  :add(nn.SelectTable(-1)) -- selecting only the last output with the 
	  :add(nn.Dropout(0.5))
	  :add(nn.Linear(hidden_size, hidden_size))
	  :add(nn.ReLU())
	  :add(nn.Linear(hidden_size, nIndex))
	  :add(nn.HardTanh())
  
  return rnn:cuda()
end

--------------------------------------------------------------------------------
function eval(i)
  rnn:evaluate()
  local predict = torch.FloatTensor(nPredict):cuda()
  local predictOneStep = torch.FloatTensor(nPredict):cuda()
  for step = 1,rho do
    predict[step] = sequence[step] -- take the first rho instances and predict the rest. 
	predictOneStep[step] = sequence[step]
  end

  local start = torch.Tensor(rho,1,nIndex):zero():cuda()
  local startOneStep = torch.Tensor(rho,1,nIndex):zero():cuda()
  
  for iteration = 0,(nPredict-rho-1) do
  
    for step = 1,rho do
      start[step] = predict:index(1,torch.LongTensor({step+iteration})):view(1,1)
	  startOneStep[step] = sequence[step + iteration]
    end
	
    rnn:forget()
    local output = rnn:forward(start)
	predict[iteration + rho + 1] = output

	-- In this case, prediction is given based on a set of perfectly correct previous values
	local outputOneStep = rnn:forward(startOneStep)
    predictOneStep[iteration + rho + 1] = outputOneStep
	
  end

  gnuplot.pngfigure("output_" .. nb_layers .. "x" .. model .. "_b" .. batch_size .. "_h" .. hidden_size .. "_i" .. i .. ".png")
  gnuplot.plot({'predictOneStep',predictOneStep,'+'} ,{'predict',predict,'+'},{'cos',sequence:narrow(1,1,nPredict),'-'})
  gnuplot.plotflush()

end

--------------------------------------------------------------------------------
function train()
  rnn = buildModel()
  rnn:training()
  print(rnn)

  offsets = {}
  for i=1,batch_size do
     table.insert(offsets, math.ceil(math.random()* (sequence:size(1)- rho) ))
  end
  offsets = torch.LongTensor(offsets)

  local train_losses = {}

  for iteration=1,iters do
    rnn:forget()
    local inputs, targets
    inputs = torch.Tensor(rho,batch_size,nIndex):zero():cuda()
	
    for j = 1,batch_size do   -- we can change it to modulo sequence:size(1) - rho
       if offsets[j] > sequence:size(1) - rho then
          offsets[j] = 1
       end
    end
    for step = 1,rho do
        inputs[step] = sequence:index(1, offsets):view(batch_size,nIndex)
        offsets:add(1)
    end
    targets = sequence:index(1, offsets)
    rnn:zeroGradParameters()
    local outputs = rnn:forward(inputs)

    local err
	err = criterion:forward(outputs, targets)
	local gradOutputs = criterion:backward(outputs, targets)
	local gradInputs = rnn:backward(inputs, gradOutputs)

    print(string.format("Iteration %d ; NLL err = %f ", iteration, err))
    train_losses[iteration] = err
    rnn:updateParameters(learning_rate)

     if iteration % eval_every == 0 or iteration == iters then
       eval(iteration)
      --  torch.save("model_" .. iteration .. ".t7", rnn)
       learning_rate = learning_rate * 0.1
     end
  end
  torch.save("train_losses_" .. nb_layers .. "x" .. model .. "_b" .. batch_size .. "_h" .. hidden_size .. "_i" .. iters .. ".txt",train_losses)
  return train_losses
end



local rnn
criterion = nn.MSECriterion():cuda()

local x = torch.linspace(0,20000, 200000)
sequence = torch.cos(x):cuda()

print("Hypertuning")
for n = 1,4 do
  for _,b in ipairs{80, 200, 500 } do
	for _,h in ipairs{300, 200, 100, 50} do
	  for _,d in ipairs{0,0.2,0.5} do
		nb_layers = n
		batch_size = b
		hidden_size = h
		dropout = d
		iters = 4000 
		eval_every = 2000
		print(nb_layers, model, "batch size: " .. batch_size, "hidden size: " .. hidden_size,"dropout: " .. dropout)
		train()
	  end
	end
  end
end