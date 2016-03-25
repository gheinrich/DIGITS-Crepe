
-- return function that returns network definition
return function(params)
    -- get number of classes from external parameters
    local nclasses = 14 -- params.nclasses or 1

    if pcall(function() require('cudnn') end) then
       print('Using CuDNN backend')
       backend = cudnn
       convLayer = cudnn.SpatialConvolution
       convLayerName = 'cudnn.SpatialConvolution'
    else
       print('Failed to load cudnn backend (is libcudnn.so in your library path?)')
       if pcall(function() require('cunn') end) then
           print('Falling back to legacy cunn backend')
       else
           print('Failed to load cunn backend (is CUDA installed?)')
           print('Falling back to legacy nn backend')
       end
       backend = nn -- works with cunn or nn
       convLayer = nn.SpatialConvolutionMM
       convLayerName = 'nn.SpatialConvolutionMM'
    end

    local alphabet_len = 69

    local net = nn.Sequential()
    -- alphabet_len x 1014
    net:add(backend.TemporalConvolution(alphabet_len, 256, 7))
    net:add(nn.Threshold())
    net:add(nn.TemporalMaxPooling(3, 3))
    -- 336 x 256
    net:add(backend.TemporalConvolution(256, 256, 7))
    net:add(nn.Threshold())
    net:add(nn.TemporalMaxPooling(3, 3))
    -- 110 x 256
    net:add(backend.TemporalConvolution(256, 256, 3))
    net:add(nn.Threshold())
    -- 108 x 256
    net:add(backend.TemporalConvolution(256, 256, 3))
    net:add(nn.Threshold())
    -- 106 x 256
    net:add(backend.TemporalConvolution(256, 256, 3))
    net:add(nn.Threshold())
    -- 104 x 256
    net:add(backend.TemporalConvolution(256, 256, 3))
    net:add(nn.Threshold())
    net:add(nn.TemporalMaxPooling(3, 3))
    -- 34 x 256
    net:add(nn.Reshape(8704))
    -- 8704
    net:add(nn.Linear(8704, 1024))
    net:add(nn.Threshold())
    net:add(nn.Dropout(0.5))
    -- 1024
    net:add(nn.Linear(1024, 1024))
    net:add(nn.Threshold())
    net:add(nn.Dropout(0.5))
    -- 1024
    net:add(nn.Linear(1024, 14))
    net:add(backend.LogSoftMax())

    -- weight initialization
    local w,dw = net:getParameters()
    w:normal():mul(5e-2)

    local function oneHotEncoder(input)
        local bs = input:size(1)
        local featureLen = input:size(2)
        local featureWidth = alphabet_len
        local output = torch.FloatTensor(bs, featureLen, featureWidth):zero()
        for n=1,bs do
            for f=1,featureLen do
                local c = input[n][f][1][1]
                if c > 0 then
                    output[n][f][c] = 1
                end
            end
        end
        return output
    end

    return {
        model = net,
        loss = nn.ClassNLLCriterion(),
        inputHook = oneHotEncoder,
        trainBatchSize = 128,
        validationBatchSize = 256,
    }
end

