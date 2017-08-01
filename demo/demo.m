%segm
cmapSegm = segmColorMap();
cmapSegm = imresize(cmapSegm,[256, 3], 'nearest');

%depth
cmapDepth = colormap('copper');
cmapDepth = [0.5 0.5 0.5;cmapDepth(end:-1:1, :)];
cmapDepth = imresize(cmapDepth,[256, 3], 'nearest');

showMatfile('samples/segm1_pred.mat', cmapSegm, 15);
showMatfile('samples/depth1_pred.mat', cmapDepth, 20);
