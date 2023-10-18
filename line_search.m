function alpha = line_search(Bd, Bx, gradf, curvf, ninner, J, cost_fun, x, s, type)

switch type
    
    case 'mm'
        % Majorize Minimize Approach for Line Search
        c1f = @(alf) sum(arrayfun(@(j) vec(real(Bd{j}))' * vec(gradf{j}(Bx{j} + alf * Bd{j})), 1:J));
        
        c2f = @(alf) sum(arrayfun(@(j) vec(real(Bd{j}))' * (curvf{j}(vec(Bx{j} + alf * Bd{j})) .* vec(Bd{j})), 1:J));
        
        alpha = 0;

        for in_iter = 1:ninner
            c1_alf = c1f(alpha);
            c2_alf = c2f(alpha);
            alpha = alpha - (c1_alf / c2_alf);
        end
        
    case 'backtrack'
        % Backtracking Line Search
        
        tau     = 0.8;      % Magic Numbers
        tau_2   = 1.3;      % 
        alpha   = 2;        % Initial Step size
        flag    = 0;
        cost_old = cost_fun(x, 0);
        
        for in_iter = 1:ninner
            
            x_temp      = x + alpha * s;
            cost_temp   = cost_fun(x_temp, 0);
            
            if cost_temp.totalCost > cost_old.totalCost && flag == 0
                alpha = alpha * tau;
            elseif cost_temp.totalCost < cost_old.totalCost
                alpha = alpha * tau_2;
                cost_old = cost_temp;
                flag = 1;
            elseif cost_temp.totalCost > cost_old.totalCost && flag == 1
                alpha = alpha / tau_2;
                return;
            elseif abs(cost_temp.totalCost - cost_old.totalCost) < 1e-5
                return;
            end
            
        end
        
end

end