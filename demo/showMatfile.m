function showMatfile(matfile, cm, N)
[dn, bn, ~] = fileparts(matfile);
imgfile = fullfile(dn, [bn '.png']);
load(matfile); % x
x = double(squeeze(x));
img = ind2rgb(uint8(255*imshow_norm(x, [1 N])), cm);
imwrite(img, cm, imgfile);
figure; imshow(uint8(255*img))

end
