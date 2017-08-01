function out = imshow_norm(img, B)
    if(nargin < 2)
        mi = min(img(:));
        ma = max(img(:));
    else
        mi = B(1);
        ma = B(2);
    end
    out = (img-mi)./(ma - mi);
end