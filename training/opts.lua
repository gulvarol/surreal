local M = { }

function M.parse(arg)
    local cmd = torch.CmdLine()
    cmd:text()
    cmd:text('Training script')
    cmd:text()
    cmd:text('Options:')
    ------------ General options --------------------
    cmd:option('-dataRoot',        paths.home ..  '/datasets/', 'Home of datasets')
    cmd:option('-logRoot',         paths.home ..  '/cnn_saves/', 'Home of datasets')
    cmd:option('-datasetname',     'cmu',         'Name of the dataset (Options: cmu')
    cmd:option('-dirName',         './',          'Experiment name')
    cmd:option('-data',            './',          'Path to train/test splits') -- set in main
    cmd:option('-save',            './save',      'Directory in which to log experiment') -- set in main
    cmd:option('-cache',           './cache',     'Directory in which to cache data info') -- set in main
    cmd:option('-plotter',         'plot',        'Path to the training curve.') -- set in main
    cmd:option('-trainDir',        'train',       'Directory name of the train data')
    cmd:option('-testDir',         'val',         'Directory name of the test data')
    cmd:option('-manualSeed',      1,             'Manually set RNG seed')
    cmd:option('-GPU',             1,             'Default preferred GPU')
    cmd:option('-nGPU',            1,             'Number of GPUs to use by default')
    cmd:option('-backend',         'cudnn',       'Backend')
    cmd:option('-verbose',         false,         'Verbose')
    cmd:option('-show',            false,         'Visualize input/output')
    cmd:option('-continue',        false,         'Continue stopped training')
    cmd:option('-evaluate',        false,         'Final predictions')
    cmd:option('-saveScores',      true,          'Score saving to txt')
    ------------- Data options ------------------------
    cmd:option('-nDonkeys',        8,             'Number of donkeys to initialize (data loading threads)') 
    cmd:option('-loadSize',        {3, 240, 320}, '(#channels, height, width) of images before crop')
    cmd:option('-inSize',          {3, 256, 256}, '(#channels, height, width) of the input')
    cmd:option('-outSize',         {64, 64},      'Ground truth dimensions') -- set in main
    cmd:option('-nOutChannels',    20,            'Number of output channels') -- set in main
    cmd:option('-extension',       {'mp4'},       'Video file extensions') -- set in main
    cmd:option('-scale',           .25,           'Degree of scale augmentation')
    cmd:option('-rotate',          30,            'Degree of rotation augmentation')
    cmd:option('-supervision',     'depth',       'Options: depth, segm')
    cmd:option('-clipsize',        100,           'Number of frames in each video clip.')
    cmd:option('-jointsIx',        {8, 5, 2, 3, 6, 9, 1, 7, 13, 16, 21, 19, 17, 18, 20, 22}, 'Joints ix')
    cmd:option('-stp',             0.045,         'Depth quantization step')
    cmd:option('-depthClasses',    19,            'Number of depth bins for quantizing depth map (odd number)')
    cmd:option('-segmClasses',     15,            'Number of segmentation classes (14 body parts, +1 for background).')
    ------------- Training options --------------------
    cmd:option('-nEpochs',         30,            'Number of total epochs to run')
    cmd:option('-epochSize',       2000,          'Number of batches per epoch')
    cmd:option('-epochNumber',     1,             'Epoch number')
    cmd:option('-batchSize',       6,             'Mini-batch size')
    ---------- Optimization options ----------------------
    cmd:option('-LR',              1e-3,          'learning rate; if set, overrides default')
    cmd:option('-momentum',        0,             'momentum')
    cmd:option('-weightDecay',     0,             'weight decay')
    cmd:option('-alpha',           0.99,          'Alpha for rmsprop')
    cmd:option('-epsilon',         1e-8,          'Epsilon for rmsprop')
    ---------- Model options ----------------------------------
    cmd:option('-netType',         'hg',          'Model type') -- set in main
    cmd:option('-retrain',         'none',        'Path to model to retrain/evaluate with')
    cmd:option('-training',        'scratch',     'Options: scratch, pretrained')
    cmd:option('-optimState',      'none',        'Path to an optimState to reload from')
    cmd:option('-nStack',          8,             'Number of stacks in hg network')
    cmd:option('-nFeats',          256,           'Number of features in the hourglass')
    cmd:option('-nModules',        1,             'Number of residual modules at each location in the hourglass')
    cmd:option('-upsample',        false,         '4 times smaller output or full resolution.')
    cmd:text()

    local opt = cmd:parse(arg or {})

    return opt
end

return M
