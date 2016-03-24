
-- return function that returns network definition
return function(params)
    -- get number of classes from external parameters
    local nclasses = params.nclasses or 1

    -- get number of channels from external parameters
    local channels = 1
    -- params.inputShape may be nil during visualization
    if params.inputShape then
        channels = params.inputShape[1]
        assert(params.inputShape[2]==28 and params.inputShape[3]==28, 'Network expects 28x28 images')
    end

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
    net:add(nn.TemporalConvolution(alphabet_len, 256, 7))
    net:add(nn.Threshold())
    net:add(nn.TemporalMaxPooling(3, 3)
    -- 336 x 256
    net:add(nn.TemporalConvolution, 256, 256, 7))
    net:add(nn.Threshold())
    net:add(nn.TemporalMaxPooling(3, 3)
    -- 110 x 256
    net:add(nn.TemporalConvolution, 256, 256, 3))
    net:add(nn.Threshold())
    -- 108 x 256
    net:add(nn.TemporalConvolution, 256, 256, 3))
    net:add(nn.Threshold())
    -- 106 x 256
    net:add(nn.TemporalConvolution, 256, 256, 3))
    net:add(nn.Threshold())
    -- 104 x 256
    net:add(nn.TemporalConvolution, 256, 256, 3))
    net:add(nn.Threshold())
    net:add(nn.TemporalMaxPooling(3, 3)
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
    net:add(nn.Linear(1024, nclasses))
    net:add(nn.Threshold())
    net:add(nn.Dropout(0.5))
    net:add(nn.LogSoftMax())

    local function oneHotEncoder(input)
    end

    return {
        model = net,
        loss = nn.ClassNLLCriterion(),
        inputHook = oneHotEncoder,
        trainBatchSize = 128,
        validationBatchSize = 128,
    }
end

