 require 'nn'
 
 -- a lookup table containing 10 tensors of size 3
 module = nn.LookupTable(10, 3) 

 input = torch.Tensor(4)
 input[1] = 1; input[2] = 2; input[3] = 1; input[4] = 10;
 print(module:forward(input))
 
 --[[
-0.2267  0.2475  0.6560
 0.8215 -1.3784 -0.6584
-0.2267  0.2475  0.6560
 0.3296 -0.9561 -0.4867
[torch.DoubleTensor of size 4x3]


Note that the first row vector is the same as the 3rd one!
]]
 
 
 