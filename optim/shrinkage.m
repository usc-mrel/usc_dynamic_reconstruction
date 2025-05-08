function T = shrinkage(x, a)
y = abs(x)-a;
y(y<0)=0;
T = exp(1i*angle(x)).*y;
% T = x-a;
% T(T<0)=0;

end