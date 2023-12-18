clear

load delay_matrices_new

S_rel = readtable("growth_rates_clustered_new.csv").S;
all_variants = readtable("growth_rates_clustered_new.csv").variant;
countries = table2array(readtable("countries_clustered.csv", 'ReadVariableNames', true));
% var_prev_S = readtable('Processed_res21_1_clusteredGT2.csv');
%var_prev_S = readtable('Processed_res21_1_clustered.csv');
var_prev_S = readtable('Processed_res21_1_clustered.csv');
var_prev_S = var_prev_S(var_prev_S.pangoLineage~="21L",:);
var_prev_GT = readtable('NEW_DATA_RetroS.csv');%var_prev_S;%readtable('splined_prevT21i_clusteredF_GT.csv');
day_zeroS = min(var_prev_S.date);
day_zeroGT = min(var_prev_GT.date);

var_prev_S.Day = days(var_prev_S.date - day_zeroS);
var_prev_GT.Day = days(var_prev_GT.date - day_zeroGT);

x=1/3;
x_list = 0.1:0.05:0.35;


%% Only keep 30 countries with most data
% val_counts = tabulate(var_prev_S.country);
% val_counts = sortrows(val_counts,-2);
% top_vals = val_counts(1:30,:);
% top = top_vals(:,1);
% % countries = top;
% var_prev_S = var_prev_S(ismember(var_prev_S.country, top),:);
% var_prev_GT = var_prev_GT(ismember(var_prev_GT.country, top),:);

% [pp, indices] = sort(S_rel,'descend');
% 
% indices = indices(1:20);
%% retrospective evaluations of base and trivial
var_list =  ["22A.Omicron"]%string(all_variants);% use all  %string(all_variants(indices)); %["BA.1"]; %list of variants i would like to retrospectively evaluate
min_points = 19; %ensures atleast 3 common variants




%for each var
for idx=1:numel(var_list)
    full_res = struct('countryData',{});

    mae_vals = [];
    median_vals = [];
    
    
    mape_vals = [];
    median_ape_vals = [];
    
    
    
    mae_vals_trivial = [];
    median_vals_trivial = [];
    
    mape_vals_trivial = [];
    median_ape_vals_trivial = [];

    %get data up till t-1 where var first becomes prevalent in a country
    v = var_list(idx);
%     var_table1 =  var_prev_S(var_prev_S.pangoLineage == (v) & var_prev_S.prev >= x & var_prev_S.count >=0.5,:);
    var_table_GT1 =  var_prev_GT(var_prev_GT.pangoLineage == (v) & var_prev_GT.prev >= x & var_prev_GT.count >=5,:);
%     var_table1 = sortrows(var_table1,'Day','ascend');
    var_table_GT1 = sortrows(var_table_GT1,'Day','ascend');
    dates = unique(var_table_GT1.date);

    best_perf = meshgrid(1:3, 1:length(countries)); %Keep track of the prev best performing source countries
    ordered_countries_GT = string(unique(var_table_GT1.country, 'stable'));

    for d=1:numel(dates)
%         var_table = var_table1(var_table1.date < dates(d),:);
        var_table_GT = var_table_GT1(var_table_GT1.date >= dates(d),:);
        day_zero = min(var_table_GT.Day);
        

        retro_table = var_prev_S(var_prev_S.Day < (day_zero-13) & var_prev_S.prev >= x,:); %& var_prev_S.pangoLineage ~= v
    %     ordered_countries = string(unique(sortrows(retro_table,'Day','ascend').country, 'stable'));
    
        if isempty(retro_table) || length(ordered_countries_GT) < 2
            v
            mae_vals(end+1) = nan;
            median_vals(end+1) = nan;

            mae_vals_trivial(end+1) = nan;
            median_vals_trivial(end+1) = nan;
            continue
        end
        %get c coefficients
        s_temp = var_prev_GT(var_prev_GT.date <= dates(d),:);
        all_vars_temp = unique(s_temp.pangoLineage);
        s_temp = sortrows(s_temp,'date','descend');
        [~, first_indices, ~] = unique(s_temp.pangoLineage, 'stable');
        % Create a logical index for the valid indices
        valid_idx = ismember(s_temp.pangoLineage(first_indices), all_vars_temp);
        % Extract the corresponding rows from the table
        resultTable = s_temp(first_indices(valid_idx), :);
        resultTable = sortrows(resultTable,'pangoLineage','ascend');
        all_vars_temp = resultTable.pangoLineage;
        S_rel = resultTable.S;
        c_vals = baseline_c(S_rel,all_vars_temp,countries,var_prev_S,x_list,min_points,v,x);
        mean_delay = trivial_m(retro_table,mats_prev,all_variants);
    
        % given it fist appears in country X, find time till it reaches another
        % next country pair (and becomes prevalent):
        
        mae_T = nan(length(countries),length(countries));
        mae_Trivial = [];
    
        mape_T = [];
        mape_Trivial = [];
        for j =1:length(ordered_countries_GT) %might need to be different for all & also other should be here (Si =~ Sj)
            for k =1:length(ordered_countries_GT)
                if j==k
                    continue
                end
%                 if k ~= 86 || j~= 82
%                     continue
%                 end
                c1 = c_vals(j,k,1);
                c2 = c_vals(j,k,2);
                
                %Retrospectively unavailable
                if (isnan(c1) || isnan(c2)) %|| (isnan(mean_delay(j,k)))
                    continue
                end
                %get list of variants in c
                % ountry j
                coj = string(ordered_countries_GT(j));  
                var_prev_j = var_table_GT(var_table_GT.country == (coj),:);
                day_zero_j = min(var_prev_j.Day);
    
                %get list of variants in country k
                cok = string(ordered_countries_GT(k));  
                var_prev_k = var_table_GT(var_table_GT.country == (cok),:);
                day_zero_k = min(var_prev_k.Day);
                
                if isempty(var_prev_k) || isempty(var_prev_j)
                    continue
                end
%                 if cok ~= "USA" || coj ~= "United Kingdom"
%                     continue
%                 end
    
                if day_zero_k < day_zero_j % it has to prevale in country j first
                    continue
                else
                    tau_gt = day_zero_k - day_zero_j;
                end
                
                Si = S_rel(strcmp(all_vars_temp,v)); 
    
                var_temp1 = var_prev_j(var_prev_j.Day == day_zero_j,:);
                indx = strcmp(all_vars_temp,var_temp1.pangoLineage2(1));
                if sum(indx)==0
                    Sj = 0;
                else
                    Sj = S_rel(indx);
                end
                
                var_temp2 = var_prev_k(var_prev_k.Day == day_zero_k,:);
                indx = strcmp(all_vars_temp,var_temp2.pangoLineage2(1));
                if sum(indx)==0
                    Sk = 0;
                else
                    Sk = S_rel(indx);
                end
                
                if Sj~=Sk %&& 
                    continue
                end

                Sij = Si - Sj;
                Sik = Si - Sk;
                Skj = Sk - Sj;
                
    
                tau = (((Skj)/(Sij*Sik))*log(x/(1-x))) - ((Skj)/(Sij*Sik))*c1  +(c2/Sij);
                mean_tau = mean_delay(j,k);
                
                tau = ceil(tau/14)*14;
                mean_tau = ceil(mean_tau/14)*14;
                AE = abs(tau - tau_gt);
    
%                 PE = AE/tau_gt;
                
                if isinf(AE)
                    continue
                end

                %Find index of target country k
                idxc = find(countries== cok);
                idxc2 = find(countries== coj);
%                 mae_T(:,end+1) = nan(size(countries));
%                 mae_T(idxc,end) = AE;
                mae_T(idxc,idxc2) = AE;


%                 mae_Trivial(idxc,end+1) = abs(mean_tau - tau_gt);

%                 mae_T = [mae_T; AE];
%                 mape_T = [mape_T; PE];
    
                mae_Trivial = [mae_Trivial; abs(mean_tau - tau_gt)];
%                 mape_Trivial = [mape_Trivial; abs(mean_tau - tau_gt)/tau_gt];
            end
        end
%         if isnan(mae_T)
%             v
%             continue
%         end
    %     %find mae and mse and save
        
%         all_vals(:,:,end+1) = mae_T; 
        full_res(d).countryData = mae_T;

        %rather than median, do a check on the last best performing country
        %pairs. Test for 5
        [~,sortedIndicies] = sort(mae_T,2);
        vals = nan(length(countries),1);
        for r=1:length(countries)
            temp = mae_T(r,best_perf(r,:)); %If atleast 1 country previously performed, use it
            if all(isnan(temp)) %if all values are nan, then use all
                vals(r) = nanmedian(mae_T(r,:),2);
            else
                vals(r) = nanmedian(temp,2);
            end
        end
        
        %These are mean/median of countries over time
        mae_vals = [mae_vals nanmean(vals)];
        median_vals = [median_vals nanmedian(vals)];

%         mae_vals = [mae_vals nanmean(nanmedian(mae_T,2))];
%         median_vals = [median_vals nanmedian(nanmedian(mae_T,2))];
        best_perf = sortedIndicies(:,1:3); %Update the best source countries


%         mape_vals = [mape_vals nanmean(mape_T)];
%         median_ape_vals = [median_ape_vals nanmedian(mape_T)];
    
        mae_vals_trivial = [mae_vals_trivial nanmean(mae_Trivial)];
        median_vals_trivial = [median_vals_trivial nanmedian(mae_Trivial)];
%         mape_vals_trivial = [mape_vals_trivial nanmean(mape_Trivial)];
%         median_ape_vals_trivial = [median_ape_vals_trivial nanmedian(mape_Trivial)];
    end  

    %save(v+"_X.mat", 'median_vals_trivial', 'mae_vals_trivial', 'median_vals', 'mae_vals', 'full_res');
end


%%
var_prev = readtable('splined_prevT21i_clusteredF_GTA.csv');
var_prev_S = readtable('splined_prevT21i_clusteredFA.csv');

for c = 83%1:numel(countries)
    country = string(countries{c});
    v_prev = var_prev(var_prev.country == country & var_prev.count > 14,:);
    v_prev_S = var_prev_S(var_prev_S.country == country & var_prev_S.count >=0.5,:); %
    var_list = unique(v_prev.pangoLineage);
    X = numel(var_list);
    colorsRGB = rand(X,3);
    color_list = cell(X,1);
    for i=1:X
        color_list{i} = rgb2hex(colorsRGB(i,:));
    end
    figure(c);
    hold on;
    for v = 1:X
        var = string(var_list{v});
        v_prevC = v_prev(v_prev.pangoLineage == var,:);
        v_prevC_S = v_prev_S(v_prev_S.pangoLineage == var,:);
        v_prevC = sortrows(v_prevC,'date','ascend');
        v_prevC_S = sortrows(v_prevC_S,'date','ascend');
        if isempty(v_prevC) || isempty(v_prevC_S) || v == 4   
            plot(datetime("2023-04-11"),0)
            plot(datetime("2023-04-11"),0,'--')
            continue
        end
        plot(v_prevC_S.date,v_prevC_S.prevsO,'Color',color_list{v});
        plot(v_prevC.date,v_prevC.prevsO, '--', 'Color',color_list{v});
    end
    ylabel('prev');
    xlabel('date');
    leg = cell(1,2*numel(var_list));
    for i = 1:numel(var_list)
        index = 2*i-1;
        leg{index} = var_list{i};
        leg{index+1} = [var_list{i} '_GT'];
    end
    legend(leg);
    hold off;
end
%%
function hex = rgb2hex(rgb)
    hex = reshape(dec2hex(round(rgb*255), 2).', 1, []);
    hex = ['#' hex];
end