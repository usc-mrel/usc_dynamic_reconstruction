function image = NUFFT_adj(kSpace_radial,N)

kSpace_radial = bsxfun(@times,N.W,kSpace_radial); % density compensation
%filter
kSpace_cart = NUFFT.rad2cart(kSpace_radial,N);

%siz = G.sx_over + G.core_size(1);
%[x,y] = meshgrid((1:siz)-siz/2,(1:siz)-siz/2);
%W = sqrt(x.^2+y.^2); W = W/max(W(:)); W = 1./W;
%kSpace_cart = bsxfun(@times,kSpace_cart,W);

%kSpace_cart(1:2:end,:,:,:) = -kSpace_cart(1:2:end,:,:,:);
%kSpace_cart(:,1:2:end,:,:) = -kSpace_cart(:,1:2:end,:,:);
kSpace_cart = fftshift(kSpace_cart,1);
kSpace_cart = fftshift(kSpace_cart,2);

image_over = ifft2(kSpace_cart);

image_over = fftshift(image_over,1);
image_over = fftshift(image_over,2);

sx = N.siz(1);
sx_over = N.sx_over;
core_size = N.core_size;

%filter = sum(abs(image_over(1:round(sx_over/4),:,:,:)),1);
%filter_scale = max(filter,2);
%filter = bsxfun(@rdivide,filter,filter_scale);
%image_over = bsxfun(@rdivide,image_over,filter);

throw_away_x = (sx_over+core_size(1)-sx)/2;
throw_away_y = (sx_over+core_size(2)-sx)/2;
throw_away_x = round(throw_away_x);
throw_away_y = round(throw_away_y);

image = image_over(throw_away_x+1:throw_away_x+sx,throw_away_y+1:throw_away_y+sx,:,:,:);

image = bsxfun(@times,image,N.Apodizer);

%image(1:2:end,:,:,:) = -image(1:2:end,:,:,:);
%image(:,1:2:end,:,:) = -image(:,1:2:end,:,:);