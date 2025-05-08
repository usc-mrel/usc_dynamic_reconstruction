 function ob = elementwise(w)

image_size = size(w);


idim = image_size ;
odim = image_size ;

forw = @(arg, x) TFD_forw(x, w);
back = @(arg, y) TFD_adj(y, w);

ob = fatrix2('idim', idim, 'odim', odim, ...
    'does_many', 1, ...
    'forw', forw, 'back', back);

 end
 
 function [y] = TFD_forw(x,w)
    y = w.*x;
 end
 
function [x] = TFD_adj(y,w)
    x = conj(w).*y;
end
 