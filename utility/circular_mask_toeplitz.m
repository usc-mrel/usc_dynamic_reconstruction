function y = circular_mask(w)
y = zeros(w, w);
for i = 2:1:w
    r = round(sqrt((w/2)^2-(w/2+1-i)^2));
    y(w/2+1-r:w/2+r, i) = 1;        
end
end