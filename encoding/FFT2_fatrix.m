 function ob = FFT2_fatrix(image_size)

arguments
    image_size 
end

idim = image_size;
odim = image_size;

forw = @(arg, x) TFD_forw(x);
back = @(arg, y) TFD_adj(y);

ob = fatrix2('idim', idim, 'odim', odim, ...
    'does_many', 1, ...
    'forw', forw, 'back', back);

 end
 
 function [y] = TFD_forw(x)
    y = fft2(x);
 end
 
function [x] = TFD_adj(y)
    x = ifft2(y);
end
 