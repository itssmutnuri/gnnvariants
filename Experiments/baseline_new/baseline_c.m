function [c_vals] = baseline_c(S_rel,all_variants,countries,var_prev_S,x_list,min_points,vO,xO)
    v_test = [];
    count = 0;
    c_vals= nan(size(countries,1),size(countries,1),2);
    for j = 1:length(countries)
%         tic;
        for k =1:length(countries)
%             if j ~= 35 || k~= 6
%                 continue
%             end
            xi_vals = [];
            tau_vals = [];
            x_vals = [];
            y_vals = [];
            % if j==k skip
            if j==k
                continue
            end
            coj = string(countries(j)); 
            cok = string(countries(k)); 
            markers = {'o', 's', 'd', '^', 'v', '>', '<', 'p', 'h'};
            colors = {'r', 'g', 'b', 'c', 'm', 'y', 'k'};
            var_legend = [];
%             figure(1);
%             hold on;
            for xi = 1:numel(x_list)
                x = x_list(xi);
                %Get retrosepctive data for given X
                var_table =  var_prev_S(var_prev_S.pangoLineage == (vO) & var_prev_S.prev >= x & var_prev_S.count >=0.5,:);
                var_table = sortrows(var_table,'Day','ascend');
                day_zero = min(var_table.Day);
                retro_table = var_prev_S(var_prev_S.Day < day_zero,:); % & var_prev_S.prev >= x % & var_prev_S.pangoLineage ~= vO


                %get list of variants in country j 
                var_prev_j = retro_table(retro_table.country == (coj),:);
                vars_j = unique(var_prev_j.pangoLineage);
    
                %get list of variants in country k 
                var_prev_k = retro_table(retro_table.country == (cok),:);
                vars_k = unique(var_prev_k.pangoLineage);
    
                %get list of variants that start in country j before k
                vars_common = intersect(vars_j,vars_k);
                %if none, continue
                if numel(vars_common) < 3%min_points
                    continue
                end
                
    %                 v_test = [v_test; numel(vars_common)];
                
                
                for i=1:numel(vars_common)  
                    v = string(vars_common{i});
                    var_temp1 = (var_prev_j(var_prev_j.pangoLineage == (v) & var_prev_j.prev >= xO & var_prev_j.count >=0.5,:));
                    var_temp2 = (var_prev_k(var_prev_k.pangoLineage == (v)  & var_prev_k.prev >= x & var_prev_k.count >=0.5,:));
                       % for prev >1/3, this insures that tau > 0
                    if (isempty(var_temp2)) || (isempty(var_temp1)) || (min(var_temp2.Day) <= min(var_temp1.Day))                       
                        continue
                    end
                    
                    
                    %if there is no other variant present in the countries,
                    %assume 0
    
    
                    Si = S_rel(strcmp(all_variants,v)); 
                    indx = strcmp(all_variants,var_temp1.pangoLineage2(1));
                    if sum(indx)==0
                        Sj = 0;
                    else
                        Sj = S_rel(indx);
                    end
    
                    indx = strcmp(all_variants,var_temp2.pangoLineage2(1));
                    if sum(indx)==0
                        Sk = 0;
                    else
                        Sk = S_rel(indx);
                    end
                    
                    Sij = Si - Sj;
                    Sik = Si - Sk;
                    Skj = Sk - Sj;
                    if Skj == 0
                        continue
                    end
                    day_zero_j = min(var_temp1.Day); 
                    day_zero_k = min(var_temp2.Day); 
                    tau = day_zero_k - day_zero_j; 
                    %tau2: Will need to interpolate
                    y = tau - ((Skj)/(Sij*Sik))*log(x/(1-x));
                    x_val = [-((Skj)/(Sij*Sik)) 1/Sij];
    
                    y_vals = [y_vals;y];
                    tau_vals = [tau_vals;tau];
                    x_vals = [x_vals; x_val];
                    xi_vals = [xi_vals; x];
                    if ~any(strcmp(v,var_legend))
                        var_legend = [var_legend;v];
                    end
                    markerIndex = find(strcmp(v,var_legend));
                    colorIndex = mod(i, length(colors)) + 1;
                    
%                     handles(xi) = plot(tau, x, 'Marker', markers{markerIndex}, 'Color', colors{markerIndex});
                end
                
            end
            
%             hold off;
%             ylabel('X');
%             xlabel('Tau');
%             legend(var_legend);
%             legend(handles, 'Point 1', 'Point 2', 'Point 3', 'Point 4', 'Point 5', ...
%             'Point 6', 'Point 7', 'Point 8', 'Point 9', 'Point 10');
            if numel(y_vals) < min_points
                continue
            end
            count = count + 1;
            [M, M_int] = regress(y_vals,x_vals);
            cj = M(1);
            cjk = M(2);
%             if j==35 && k== 6
                scatter3(x_vals(:,1),x_vals(:,2),y_vals, 100);
                xlabel("Z_1", 'FontSize', 18);
                ylabel("Z_2", 'FontSize', 18);
                zlabel("Y", 'FontSize', 18);
                set(gca, 'FontSize', 16);
%             end
%             M = fitlm(x_vals,y_vals, 'Intercept', false);
%             cj = M.Coefficients(1,:).Estimate;
%             cjk = M.Coefficients(2,:).Estimate;
            if any(M_int(:,2) - M_int(:,1) > 200) && (Sj ~= Sk)
                continue
            end
            c_vals(j,k,1) = cj;
            c_vals(j,k,2) = cjk;
        end
%         toc;
    end
%     hist(v_test);
end