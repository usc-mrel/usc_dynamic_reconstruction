function N = init(kx,ky,over_sampling,core_size)
% N = init(kx,ky,over_sampling,core_size)

sx  = size(kx,1);

skx = size(kx,1);
nor = size(kx,2);
nof = size(kx,3);

sx_over = round(sx*over_sampling);

kx = kx*over_sampling;
ky = ky*over_sampling;

x_cart = zeros(skx,nor,nof,core_size(1));
y_cart = zeros(skx,nor,nof,core_size(2));

if mod(core_size(1),2) == 0
    x_cart(:,:,:,1) = floor(kx) - core_size(1)/2 + 1;
else
    x_cart(:,:,:,1) = round(kx) - (core_size(1)-1)/2;
end
add_x(1,1,1,:) = 1:core_size(1)-1;
x_cart(:,:,:,2:end) = bsxfun(@plus,x_cart(:,:,:,1),add_x);

if mod(core_size(2),2) == 0
    y_cart(:,:,:,1) = floor(ky) - core_size(2)/2 + 1;
else
    y_cart(:,:,:,1) = round(ky) - (core_size(2)-1)/2;
end
add_y(1,1,1,:) = 1:core_size(2)-1;
y_cart(:,:,:,2:end) = bsxfun(@plus,y_cart(:,:,:,1),add_y);

x_cart = repmat(x_cart,1,1,1,core_size(2));
x_cart = reshape(x_cart,skx,nor,nof,prod(core_size));
y_cart = permute(y_cart,[1,2,3,5,4]);
y_cart = repmat(y_cart,1,1,1,core_size(1));
y_cart = reshape(y_cart,skx,nor,nof,prod(core_size));

dx = round((x_cart - kx)*100)/100;
dy = round((y_cart - ky)*100)/100;

if mod(core_size(1),2) == 0
    x_cart = x_cart+sx_over/2+core_size(1)/2;
else
    x_cart = x_cart+sx_over/2+core_size(1)/2+0.5;
end
if mod(core_size(2),2) == 0
    y_cart = y_cart+sx_over/2+core_size(1)/2;
else
    y_cart = y_cart+sx_over/2+core_size(1)/2+0.5;
end

if mod(core_size(1),2) == 0
    indx = sub2ind([(sx_over+core_size(1)),(sx_over+core_size(2)),nof],x_cart,y_cart);
else
    indx = sub2ind([(sx_over+core_size(1)+1),(sx_over+core_size(2)+1),nof],x_cart,y_cart);
end

rad2cart = cell(1,nof);
rad_num = (1:skx*nor*prod(core_size)).';


for i=1:nof
    indx_temp = indx(:,:,i,:);
    indx_temp = indx_temp(:);
    if mod(core_size(1),2) == 0
        rad2cart{i} = sparse(indx_temp,rad_num,1,(sx_over+core_size(1))*(sx_over+core_size(2)),skx*nor*prod(core_size));
    else
        rad2cart{i} = sparse(indx_temp,rad_num,1,(sx_over+core_size(1)+1)*(sx_over+core_size(2)+1),skx*nor*prod(core_size));
    end

    clear indx_temp
end

%%% kaiser bessel
alpha = 2.34 * core_size(1);
kb = @(k,J)kaiser_bessel(k,J,alpha,0);
kb_weight_x = feval(kb,dx,core_size(1));
kb_weight_y = feval(kb,dy,core_size(2));
weight_kb = kb_weight_x.*kb_weight_y;
%weight_kb = permute(weight_kb,[1,2,4,3]);
weight_kb = reshape(weight_kb,sx*nor,nof,prod(core_size));
N.weight_kb = weight_kb;

NC = (0:sx-1)'-(sx-1)/2;
tmpx = 1 ./ kaiser_bessel_ft(NC/sx_over, core_size(1), alpha, 0, 1);
tmpy = 1 ./ kaiser_bessel_ft(NC/sx_over, core_size(2), alpha, 0, 1);
N.Apodizer = tmpx * tmpy';

%phase_scale = 1i * pi /over_sampling;
%phase = exp(phase_scale * (dx + dy));
%N.phase = reshape(phase,sx*nor,nof,prod(core_size));
%%%

N.rad2cart = rad2cart;
N.core_size = core_size;
N.sx_over = sx_over;
N.siz = [skx nor nof];

N.W = density_comp_area(kx,ky,mod(atan(ky(1,:,:)./kx(1,:,:)),pi));
%N.W = designFilter(sx,-1,'ram-lak');
%N.W(144) = N.W(144)/2;

%G.W = ones(sx,1);
