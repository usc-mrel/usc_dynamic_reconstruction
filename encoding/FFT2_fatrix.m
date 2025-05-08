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
    y = fft2(x)/sqrt(size(x,1)*size(x,2)); % Make FFT Orthogonal
 end
 
function [x] = TFD_adj(y)
    x = ifft2(y)*sqrt(size(y,1)*size(y,2)); % Make iFFT Orthogonal
end
 