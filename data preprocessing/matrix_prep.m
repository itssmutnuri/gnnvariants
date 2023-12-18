% Read the csv file into a table
clear;
ground_truth = readtable("splined_prevT21i_clusteredF_GTA");
var_prev_S = readtable("splined_prevT21i_clusteredFA");%splined_prevT21i_clusteredFA");%readtable('splined_prevTF21i.csv');
countries =table2array(readtable("countries_clustered.csv", 'ReadVariableNames', false));
var_prev = var_prev_S;

% Assign the sample countries
sample_countries = countries; %["USA", "United Kingdom", "China", "Australia", "India"] #S

% Get all unique variants
all_variants = readtable("all_vars21_clustered_new.csv").all_variants;%unique(var_prev.pangoLineage);
% Convert cell array to table
% mytable = cell2table(all_variants);
% 
% % Write table to CSV file
% writetable(mytable, 'all_vars21_clustered.csv');
% mytable = cell2table(countries);
% writetable(mytable, 'countries_clustered.csv');

% Convert date column to datetime format and compute the number of days from the minimum date
% var_prev_S = readtable('splined_prevTF21i_clusteredF.csv');%readtable('splined_prevT2.csv');
var_prev_S.date = datetime(var_prev_S.date, 'InputFormat', 'yyyy-MM-dd');
day_zeroS = min(var_prev_S.date);

var_prev.date = datetime(var_prev.date, 'InputFormat', 'yyyy-MM-dd');
day_zero = min(var_prev.date);
%CAN ADD LOOP HERE TO FIND DATE OF FIRST OCCURENCE FOR EACH VARIANT PER
%COUNTRY. THEN DAYS TILL PREV = DAY2 - DAY1



var_prev.Day = days(var_prev.date - day_zero);
var_prev_S.Day = days(var_prev_S.date - day_zeroS);
%%
ground_truth = readtable("var_data_GTOG");
sample_vars = all_variants([1 2 7:8 11 19:20 22]);%all_variants([1 2 7:8 11 19:20 22]); %0 22 23]);
% figure;
ground_truth = ground_truth(ground_truth.date<"2022-01-04",:);
c="United Kingdom";
var_legend = [];
count = 1;
% subplot(1,2,1);
figure(1)
temp_table = ground_truth((ground_truth.pangoLineage == "21J.Delta")&(ground_truth.country == c),:);
temp_table = sortrows(temp_table,'date','ascend');
temp_table2 = ground_truth((ground_truth.pangoLineage == "20I.Alpha.V1")&(ground_truth.country == c),:);
temp_table2 = sortrows(temp_table2,'date','ascend');    
com_dates = intersect(temp_table.date,temp_table2.date);

selectedRows = temp_table2(ismember(temp_table2.date, com_dates) & ismember(temp_table.date, com_dates), :);
selectedRows = selectedRows(19:23,:).prevsO;

selectedRows2 = temp_table(ismember(temp_table2.date, com_dates) & ismember(temp_table.date, com_dates), :);
selectedRows2 = selectedRows2(19:23,:).prevsO;
% if ~isempty(temp_table)
plot(com_dates(19:23,:), smooth(log(selectedRows2./selectedRows)), '-o','MarkerSize',10, 'LineWidth',3); hold on;
%     var_legend = [var_legend;var];
% end
% var;
% end
hold off;
title("21J.Delta vs 20I.Alpha.V1 (United Kingdom)");
ylabel('log(r_{ij})', 'FontSize', 20);
xlabel('date', 'FontSize', 20);
set(gca, 'FontSize', 18);
% subplot(1,2,2);
figure(2)
temp_table = ground_truth((ground_truth.pangoLineage == "21J.Delta")&(ground_truth.country == c),:);
temp_table = sortrows(temp_table,'date','ascend');
temp_table2 = ground_truth((ground_truth.pangoLineage == "21K.Omicron")&(ground_truth.country == c),:);
temp_table2 = sortrows(temp_table2,'date','ascend');    
com_dates = intersect(temp_table.date,temp_table2.date);

selectedRows = temp_table2(ismember(temp_table2.date, com_dates) & ismember(temp_table.date, com_dates), :);
selectedRows = selectedRows(35:40,:).prevsO;

selectedRows2 = temp_table(ismember(temp_table2.date, com_dates) & ismember(temp_table.date, com_dates), :);
selectedRows2 = selectedRows2(35:40,:).prevsO;
% if ~isempty(temp_table)
plot(com_dates(35:40,:), smooth(log(selectedRows./selectedRows2)), '-o','MarkerSize',10, 'LineWidth',3); hold on;
%     var_legend = [var_legend;var];
% end
% var;
% end
hold off;
title("21K.Omicron vs 21J.Delta (United Kingdom)");
ylabel('log(r_{ij})', 'FontSize', 20);
xlabel('date', 'FontSize', 20);
set(gca, 'FontSize', 18);
% legend(var_legend);
%%
ground_truth = readtable("var_data_GTOG");
sample_vars = all_variants([1 2 7:8 11 19:20 22]);%all_variants([1 2 7:8 11 19:20 22]); %0 22 23]);
figure;
ground_truth = ground_truth(ground_truth.date<"2022-01-04",:);
c="United Kingdom";
var_legend = [];
count = 1;
subplot(2,1,1);
for v = 1:length(sample_vars)
    var= string(sample_vars{v});
    temp_table = ground_truth((ground_truth.pangoLineage == var)&(ground_truth.country == c),:);
    temp_table = sortrows(temp_table,'date','ascend');       
    if ~isempty(temp_table)
        plot(temp_table.date,(temp_table.prevsO), 'LineWidth',3); hold on;
        var_legend = [var_legend;var];
    end
    var;
end
hold off;
title("United Kingdoms");
ylabel('p_i');
xlabel('date');
legend(var_legend);


subplot(2,1,2);
for v = 1:length(sample_vars)
    var= string(sample_vars{v});
    temp_table = ground_truth((ground_truth.pangoLineage == var)&(ground_truth.country == c),:);
    temp_table = sortrows(temp_table,'date','ascend');       
    if ~isempty(temp_table)
        plot(temp_table.date,log(smooth(temp_table.prev)), 'LineWidth',3); hold on;
        var_legend = [var_legend;var];
    end
    var;
end
hold off;
title(c);
ylabel('p_i');
xlabel('date');


%%

c="Sweden";
var_legend = [];
count = 1;
subplot(2,1,2);
for v = 1:length(sample_vars)
    var= string(sample_vars{v});
    temp_table = ground_truth((ground_truth.pangoLineage == var)&(ground_truth.country == c),:);
    temp_table = sortrows(temp_table,'date','ascend');       
    if ~isempty(temp_table)
        plot(temp_table.date,((temp_table.prevsO)), 'LineWidth',3); hold on;
        var_legend = [var_legend;var];
    end
    var;
end
hold off;
title(c);
ylabel('p_i');
xlabel('date');
legend(var_legend);
count = count+1;
%%
%Anything that reaches 25% (prev=1/3) is dominant.
% var_prev_D = table; %This is a table showing for each variant/country 
% for c =1:length(countries)
%     co = string(countries(c));
%     var_prev_c = var_prev_S(var_prev_S.country == (co),:);
%     for i =1:length(all_variants)
%         v = string(all_variants(i));
%         temp_table = var_prev_c(var_prev_c.pangoLineage == (v),:);
%         temp_table = sortrows(temp_table,'Day','ascend');
%         
%         day_zero = (temp_table.Day);
%         idx = find(temp_table{:,"prev"}>1/3,1);
%         if isempty(idx)
%             day_prev = min(temp_table.Day);
%         else
%             day_prev = temp_table.Day(idx);
%         end
%         temp_table.days_to_prev = max(day_prev - day_zero,0);
% 
%         var_prev_D = [var_prev_D;temp_table];
%     end
% end
% writetable(var_prev_D,"Processed_res21.csv")
% 5/15/25/50
% 10/20/33/50: (1/9)/0.25/0.5/1
%. So let us run it for r = (1/9)/0.25/0.5/1
%% Relative to Country it first appeared Anything that reaches 25% (prev=1/3) is dominant.
var_prev_D = table; %This is a table showing for each variant/country 
var_prev_B = [];

% S_rel = nan(length(sample_countries),length(all_variants),length(all_variants));
% S_rel(i,c,c) = 1; %Relative growth = 1;
for c =1:length(all_variants) 
    v = string(all_variants(c));
    temp_table = var_prev_S(var_prev_S.pangoLineage == (v),:);

    sorted_table = sortrows(temp_table,'Day','ascend');
    idx = find(sorted_table{:,"prev"}>1/3 & sorted_table{:,"count"}>1/2,1);
    if isempty(idx)
        continue %This means it never reaches prevalence anywhere %min(temp_table.Day);
    else
        day_zero = min(sorted_table.Day(idx)); 
        date_1 = min(sorted_table.date(idx));
        cou = string(sorted_table.country(idx));
    end    
    
    %sort countries variant appears in by day it becomes prevalent
    %only use the first 50% of countries it becomes prevalent in
    sorted_table = sorted_table(sorted_table.prev>1/3 & sorted_table{:,"count"}>1/2,:);
    variant_countries = unique(sorted_table.country, 'stable');
    variant_countries = variant_countries(1:ceil(numel(variant_countries)/2));

    for i =1:length(variant_countries)
        
        co = string(variant_countries(i));
        var_prev_c = temp_table(temp_table.country == (co) & temp_table{:,"count"}>1/2,:);
        if co== cou
            var_prev_c = sortrows(var_prev_c,'Day','ascend');
            var_prev_c.days_to_prev = max(day_zero - var_prev_c.Day,0);
            var_prev_D = [var_prev_D;var_prev_c];
            var_prev_B = [var_prev_B; repmat(1,size(var_prev_c,1),1)];
            continue
        end
        if size(var_prev_c,1) == 0
            continue
        end
        var_prev_c = sortrows(var_prev_c,'Day','ascend');
%         day_all = (var_prev_c.Day);
        idx = find(var_prev_c{:,"prev"}>1/3 & var_prev_c{:,"count"}>1/2,1);

        if isempty(idx) %wont be the case anymore since we choose first 50% of countries it becomes prevalent in
            day_prev = repmat(10000 + day_zero,size(var_prev_c,1),1); %This means it never reaches prevalence in this country %min(temp_table.Day);
            var_prev_c.days_to_prev = max(day_prev - day_zero,0);
            var_prev_B = [var_prev_B; repmat(0,size(var_prev_c,1),1)];
%         elseif  var_prev_c.Day(1) < var_prev_c.Day(idx)
%             day_prev = var_prev_c.Day(idx);
        else
            date_2 = var_prev_c.Day(1); %date of first appearance in country
            % fillers
            row_0 = var_prev_c(1,:);
            row_0.date = date_1; 
            row_0.count = 0;
            row_0.prev = 0;
%             if date_2<day_zero %Case testing (comment if not testing)
%                 date_2; 
%             end
            row_0.Day = min(date_2,day_zero); %This basically gives days leading up to prev

            day_prev = var_prev_c.Day(idx); %first day of prev in this country
            prevs = (day_prev-min(date_2,day_zero)):-1:1; %this gets the day diff count down from first ever appearance
            day_all = min(date_2,day_zero):day_prev-1;
            dates = flip(date_1+prevs'-1);%date_1:date_2-1;
            row_0 = repmat(row_0,size(day_all,2),1);
            row_0.date = dates;
            row_0.Day = day_all';
            row_0.days_to_prev = prevs';
            var_prev_c.days_to_prev = repmat(0,size(var_prev_c,1),1);
            if  var_prev_c.Day(1) < var_prev_c.Day(idx) %Deal with overlaps
%                 row_0.count(end-:end) = ; 
%                 row_0.prev(end-:end) = ;
%                 try
%                     x = var_prev_c.Day(idx) - var_prev_c.Day(1);
%                     row_0 = row_0(1:end-x,:);
%                 catch
%                     x = var_prev_c.Day(idx) - var_prev_c.Day(1);
%               idx is where it first becomes prevalant.
               idx2 = find(row_0{:,"Day"}==date_2,1);
               %change days till prev in overlapping area
               var_prev_c.days_to_prev(1:idx-1) =row_0.days_to_prev(end-size(var_prev_c(1:idx-1,:),1)+1:end);
               row_0 = row_0(1:idx2-1,:);
%                var_prev_c.Day(1:idx-1) =row_0.Day(end-size(var_prev_c(1:idx-1,:),1)+1:end);
%                var_prev_c.date(1:idx-1) =row_0.date(end-size(var_prev_c(1:idx-1,:),1)+1:end);
%                var_prev_c.days_to_prev(1:idx-1) =row_0.days_to_prev(end-size(var_prev_c(1:idx-1,:),1)+1:end);
%                row_0 = row_0(1:end-size(var_prev_c(1:idx-1,:),1),:);
%                row_0;
               
            end
            
            var_prev_c = [row_0; var_prev_c];
            var_prev_B = [var_prev_B; repmat(1,size(var_prev_c,1),1)];
            %%dates after necessary? probz not
        end        
        

        var_prev_D = [var_prev_D;var_prev_c];
        
    end
end
% Filter out variants which die out (days till prev = 0 AND prevalence
% becomes < 1/3
indx = (var_prev_D.days_to_prev== 0 & var_prev_D.prev < 1/3); %SHOULD BE CUTOFF ONY IF ALL COUNTRIES FOLLOW THIS RULE, NOT FOR EACH
% var_prev_B = var_prev_B(~indx,:);
var_prev_D.B = var_prev_B;
var_prev_D = var_prev_D(~indx,:);

writetable(var_prev_D,"Processed_res21_1_clustered22.csv");

%% prep GT data
var_prev_D = table; %This is a table showing for each variant/country 
var_prev_B = [];
thresh = 1/3;%r = (1/9)/0.25/0.5/1
% S_rel = nan(length(sample_countries),length(all_variants),length(all_variants));
% S_rel(i,c,c) = 1; %Relative growth = 1;
for c =1:length(all_variants) 
    v = string(all_variants(c));
    temp_table = var_prev_S(var_prev_S.pangoLineage == (v),:);

    sorted_table = sortrows(temp_table,'Day','ascend');
    idx = find(sorted_table{:,"prev"}>thresh & sorted_table{:,"count"}>14,1);
    if isempty(idx)
        continue %This means it never reaches prevalence anywhere %min(temp_table.Day);
    else
        day_zero = min(sorted_table.Day(idx)); 
        date_1 = min(sorted_table.date(idx));
        cou = string(sorted_table.country(idx));
    end    
    
    %sort countries variant appears in by day it becomes prevalent
    %only use the first 50% of countries it becomes prevalent in
    sorted_table = sorted_table(sorted_table.prev>thresh & sorted_table{:,"count"}>14,:);
    variant_countries = unique(sorted_table.country, 'stable');
    variant_countries = variant_countries(1:ceil(numel(variant_countries)/2));

    for i =1:length(variant_countries)
        
        co = string(variant_countries(i));
        var_prev_c = temp_table(temp_table.country == (co) & temp_table{:,"count"}>14,:);
        if co== cou
            var_prev_c = sortrows(var_prev_c,'Day','ascend');
            var_prev_c.days_to_prev = max(day_zero - var_prev_c.Day,0);
            var_prev_D = [var_prev_D;var_prev_c];
            var_prev_B = [var_prev_B; repmat(1,size(var_prev_c,1),1)];
            continue
        end
        if size(var_prev_c,1) == 0
            continue
        end
        var_prev_c = sortrows(var_prev_c,'Day','ascend');
%         day_all = (var_prev_c.Day);
        idx = find(var_prev_c{:,"prev"}>thresh & var_prev_c{:,"count"}>14,1);

        if isempty(idx) %wont be the case anymore since we choose first 50% of countries it becomes prevalent in
            day_prev = repmat(10000 + day_zero,size(var_prev_c,1),1); %This means it never reaches prevalence in this country %min(temp_table.Day);
            var_prev_c.days_to_prev = max(day_prev - day_zero,0);
            var_prev_B = [var_prev_B; repmat(0,size(var_prev_c,1),1)];
%         elseif  var_prev_c.Day(1) < var_prev_c.Day(idx)
%             day_prev = var_prev_c.Day(idx);
        else
            date_2 = var_prev_c.Day(1); %date of first appearance in country
            % fillers
            row_0 = var_prev_c(1,:);
            row_0.date = date_1; 
            row_0.count = 0;
            row_0.prev = 0;
%             if date_2<day_zero %Case testing (comment if not testing)
%                 date_2; 
%             end
            row_0.Day = min(date_2,day_zero); %This basically gives days leading up to prev

            day_prev = var_prev_c.Day(idx); %first day of prev in this country
            prevs = (day_prev-min(date_2,day_zero)):-14:1; %this gets the day diff count down from first ever appearance
            day_all = min(date_2,day_zero):14:day_prev-1;
            dates = flip(date_1+prevs'-14);%date_1:date_2-1;
            row_0 = repmat(row_0,size(day_all,2),1);
            row_0.date = dates;
            row_0.Day = day_all';
            row_0.days_to_prev = prevs';
            var_prev_c.days_to_prev = repmat(0,size(var_prev_c,1),1);
            if  var_prev_c.Day(1) < var_prev_c.Day(idx) %Deal with overlaps
%                 row_0.count(end-:end) = ; 
%                 row_0.prev(end-:end) = ;
%                 try
%                     x = var_prev_c.Day(idx) - var_prev_c.Day(1);
%                     row_0 = row_0(1:end-x,:);
%                 catch
%                     x = var_prev_c.Day(idx) - var_prev_c.Day(1);
%               idx is where it first becomes prevalant.
               idx2 = find(row_0{:,"Day"}==date_2,1);
               %change days till prev in overlapping area
               var_prev_c.days_to_prev(1:idx-1) =row_0.days_to_prev(end-size(var_prev_c(1:idx-1,:),1)+1:end);
               row_0 = row_0(1:idx2-1,:);
%                var_prev_c.Day(1:idx-1) =row_0.Day(end-size(var_prev_c(1:idx-1,:),1)+1:end);
%                var_prev_c.date(1:idx-1) =row_0.date(end-size(var_prev_c(1:idx-1,:),1)+1:end);
%                var_prev_c.days_to_prev(1:idx-1) =row_0.days_to_prev(end-size(var_prev_c(1:idx-1,:),1)+1:end);
%                row_0 = row_0(1:end-size(var_prev_c(1:idx-1,:),1),:);
%                row_0;
               
            end
            
            var_prev_c = [row_0; var_prev_c];
            var_prev_B = [var_prev_B; repmat(1,size(var_prev_c,1),1)];
            %%dates after necessary? probz not
        end        
        

        var_prev_D = [var_prev_D;var_prev_c];
        
    end
end
% Filter out variants which die out (days till prev = 0 AND prevalence
% becomes < 1/3
indx = (var_prev_D.days_to_prev== 0 & var_prev_D.prev < thresh);
% var_prev_B = var_prev_B(~indx,:);
var_prev_D.B = var_prev_B;
var_prev_D = var_prev_D(~indx,:);

writetable(var_prev_D,"NEW_DATA_1_9th.csv");
%% 
% prepare matrix for which variant is present in each country and a label
% if it becomes dominant in such a country
variant_matrix = zeros(length(sample_countries), length(all_variants));
variant_matrix_D = zeros(length(sample_countries), length(all_variants));
for i = 1:length(sample_countries)
    var_prev_temp = var_prev(string(var_prev.country)==sample_countries{i},:);
    var_prev_temp_D = var_prev((string(var_prev.country)==sample_countries{i}) & (var_prev.prev >= 1/3),:);
    for j = 1:length(all_variants)
    % Check if the variant is present in the country
        variant_present = sum(string(var_prev_temp.pangoLineage) == (all_variants{j}))>0;
        variant_dominant = (sum(string(var_prev_temp_D.pangoLineage) == (all_variants{j}))>0);
        variant_matrix(i, j) = variant_present;
        variant_matrix_D(i, j) = variant_dominant;
    end
end
writetable(array2table(variant_matrix),"present.csv")
writetable(array2table(variant_matrix_D),"dominant.csv")

%% delay matrices
% Initialize the mats_prev 3D matrix
mats_prev = NaN(length(sample_countries), length(sample_countries), length(all_variants));
% Loop over the sample countries to fill the mats_prev matrix
for i = 1:length(sample_countries)
    tic;
    for j = 1:length(sample_countries)
%         tic;
        if i == j
            mats_prev(i, j, :) = 0; % set them all to 0
            continue
        end

        if j < i
            mats_prev(i, j, :) = -mats_prev(j, i, :);
            continue
        end
        
        sub_vars = all_variants((variant_matrix(i,:) & variant_matrix(j,:)));
        var_countries = var_prev((string(var_prev.country) == sample_countries{i}) | (string(var_prev.country) == sample_countries{j}),:);
        for v = 1:length(sub_vars)
            var_name = sub_vars{v};
            var_table_source = var_countries((string(var_countries.pangoLineage) == var_name) & (string(var_countries.country) == sample_countries{i}) & var_countries.prev >= (1/3), :);
            var_table_dest = var_countries((string(var_countries.pangoLineage) == var_name) & (string(var_countries.country) == sample_countries{j}) & var_countries.prev >= (1/3), :);
            %Check if it doesn't become prevalent in one of the 2 countries
            if(isempty(var_table_dest) || isempty(var_table_source))
                continue
            end
            mats_prev(i, j, ismember(all_variants,var_name)) = min(var_table_dest.Day) - min(var_table_source.Day);
        end
%         toc;
    end
    toc;
end

% Display the mean and standard deviation of the mats_prev matrix
fprintf('Mean: %f\n', mean(mats_prev, 3, 'omitnan'));
fprintf('std: %f\n', std(mats_prev, 0, 3, 'omitnan'));

% Save the mats_prev matrix to a .mat file
save('delay_matrices_new.mat', 'mats_prev');
writetable(array2table(nanmean(mats_prev,3)),"mean_delay.csv");
writetable(array2table(nanmedian(mats_prev,3)),"median_delay.csv");
writetable(array2table(nanstd(mats_prev,[],3)),"std_delay.csv");
%% Find relative S values retrospectively
ground_truth = readtable("NEW_DATA.csv"); %NEW_DATA_1_9th
ground_truth.S = nan(height(ground_truth),1);
dates = unique(ground_truth.date);
thresh = 1/3;
for d=length(dates)
    % all_variants = unique(var_prev.pangoLineage);
    var_prev = var_prev_S(var_prev_S.date <= dates(d), :);
    idx1 = ground_truth.date <= dates(d);
    gt_temp = ground_truth(idx1,:);
    sample_countries = unique(gt_temp.country);%countries;
    all_variants = unique(gt_temp.pangoLineage);
    S_rel = nan(length(sample_countries),length(all_variants),length(all_variants));
    % S_rel(i,c,c) = 1; %Relative growth = 1;
     %["USA", "United Kingdom", "China", "Australia", "India"] #S
    %var_prev.date = datenum(var_prev.date);
    day_zero = min(var_prev.date);
    var_prev.Day = var_prev.date - day_zero;
    % to 2nd most dominant
    % all_variants = [all_variants[all_variants.index('B.1.1')]]
    %17 for first variant (19A)
    %Matric will be SxSxNbOfVariants prev ratio>1?
    % mats_prev = np.empty((len(sample_countries),len(sample_countries),len(all_variants)))
    % S = nan(length(sample_countries),length(all_variants));
    % S2 = np.empty((len(sample_countries),len(all_variants)))
    variant_samples = all_variants;%["20I.Alpha.V1",,];
    for i = 1:length(sample_countries)
        for j = 1:length(variant_samples)
            
            tic;
            v = variant_samples(j);


            var_table_source = var_prev( (string(var_prev.pangoLineage) == v) & (string(var_prev.country) == sample_countries{i}) & var_prev.prev>=(thresh), :);
            if height(var_table_source) < 2
                S_rel(i,j,:) = NaN;
                continue;
            end
            rel_v = unique(var_table_source.pangoLineage2);
            for v = 1:numel(rel_v)
                
                var_table_source2 = var_table_source(var_table_source.pangoLineage2 == string(rel_v{v}),:);
                ki = find(strcmp(variant_samples, rel_v{v}));
    %             if ki ~= 8 &&  ki ~=20 &&  ki ~=22
    %                 continue;
    %             end
                var_table_source2 = sortrows(var_table_source2, 'date');
                x_values = linspace(1,length(var_table_source2.prev),length(var_table_source2.prev));
                x_values = var_table_source2.date;
                % y = var_table_source.prev
                %Smooth curve
                y = smooth(var_table_source2.prev, ceil(length(var_table_source2.prev)/10), 'moving');
                if (variant_samples(j) ~= "21J.Delta" ) || (string(rel_v{v}) ~= "20I" + ...
                        ".Alpha.V1")%)
                    continue
                end
                if (string(sample_countries{i}) ~= "USA") && (string(sample_countries{i}) ~= "Sweden") && (string(sample_countries{i}) ~= "United Kingdom")
                    continue
                end
                figure(1)
                plot(x_values,log(y));
                xlabel("date");
                ylabel("log(r_{ij})"); %Belg or Canada, Germany, Croatia, Estonia, mebe finland, 
                title(variant_samples(j)+" vs " +string(rel_v{v})+ " in " + sample_countries{i});
                %Find till where xval is increasing
                continue; %uncomment the datenum if comenting this
                der = diff(y);
                c = 2;
                l=1;
                % flag = 0;
                for k = 1:length(der)
                    if der(k) < 0
                        if (c-l) < 5
                            l = l+1;
                            c = c+1;
                            continue;
                            % flag = 1;
                            % S(i,j) = NaN;
                        end
                        break;
                    end
                    c = c + 1;
                end
                %If never increasing, then it was identified late. We can remove this?
                if (c-l)<5
                    S_rel(i,j,ki) = NaN;
                    continue;
                end
                coeffs = polyfit(x_values(l:min(c,end)), log(y(l:min(c,end))), 1); %can do multinomial regression isteaad
                S_rel(i,j,ki) = coeffs(1);
            end
            
            % coeffs = np.polyfit(x_values[l:c], y[l:c], 1)
            % S2(i,j) = coeffs(1);
            toc;
        end
    end
    % 197x426x426
    S_rel(S_rel<0) = 0;
    run('get_global_s.m');
    idx1 = ground_truth.date == dates(d);
    [idx2,idx3] = ismember(ground_truth.pangoLineage, T.variant);
%     idx3 = ismember(T.variant,ground_truth.pangoLineage);
    logicIdx = idx3>0;
    logicIdx(logicIdx) = true;
    temp_vals = nan(size(idx3));
    temp_vals(logicIdx) =  T.S(idx3(logicIdx));
    ground_truth.S(idx1&idx2) = temp_vals(idx1&idx2); %T.S(idx3);

end
writetable(ground_truth,"NEW_DATA_RetroS_1_9th.csv");
%save('objs21N_rel_clustered.mat', 'S_rel');
%% Find S values
sample_countries = countries; %["USA", "United Kingdom", "China", "Australia", "India"] #S
all_variants = unique(var_prev.pangoLineage);
var_prev.date = datenum(var_prev.date);
day_zero = min(var_prev.date);
var_prev.Day = var_prev.date - day_zero;
% all_variants = [all_variants[all_variants.index('B.1.1')]]
%17 for first variant (19A)
%Matric will be SxSxNbOfVariants prev ratio>1?
% mats_prev = np.empty((len(sample_countries),len(sample_countries),len(all_variants)))
S = nan(length(sample_countries),length(all_variants));
% S2 = np.empty((len(sample_countries),len(all_variants)))
for i = 1:length(sample_countries)
    for j = 1:length(all_variants)
        tic;
        v = all_variants(j);
        var_table_source = var_prev( (string(var_prev.pangoLineage) == v) & (string(var_prev.country) == sample_countries{i}) & var_prev.prev>=(1/3), :);
        if height(var_table_source) < 2
            S(i,j) = NaN;
            continue;
        end
        var_table_source = sortrows(var_table_source, 'date');
        x_values = linspace(1,length(var_table_source.prev),length(var_table_source.prev));
        % y = var_table_source.prev
        %Smooth curve
        y = smooth(var_table_source.prev, ceil(length(var_table_source.prev)/10), 'moving');
        %Find till where xval is increasing
        der = diff(y);
        c = 2;
        l=1;
        % flag = 0;
        for k = 1:length(der)
            if der(k) < 0
                if (c-l) < 5
                    l = l+1;
                    c = c+1;
                    continue;
                    % flag = 1;
                    % S(i,j) = NaN;
                end
                break;
            end
            c = c + 1;
        end
        %If never increasing, then it was identified late. We can remove this?
        if (c-l)<5
            S(i,j) = NaN;
            continue;
        end
        coeffs = polyfit(x_values(l:min(c,end)), log(y(l:min(c,end))), 1); %can do multinomial regression isteaad
        S(i,j) = coeffs(1);
        % coeffs = np.polyfit(x_values[l:c], y[l:c], 1)
        % S2(i,j) = coeffs(1);
        toc;
    end
end
% 197x426x426
S(S<0) = 0;

save('objs21N.mat', 'S');
%% normalize by country S
m = mats_prev;
mats_dyn = NaN(length(sample_countries),length(sample_countries),length(all_variants));
for i = 1:length(sample_countries)
    s_i = S(i,:);
    for j = 1:length(sample_countries)
        mats_dyn(i,j,:) = squeeze(m(i,j,:)).*s_i';
    end
end

writetable(array2table(nanmean(mats_dyn,3)),"mean_delay_S_country.csv");
writetable(array2table(nanmedian(mats_dyn,3)),"median_delay_S_country.csv");
writetable(array2table(nanstd(mats_dyn,[],3)),"std_delay_S_country.csv");

%% normalize by global S
S_global = table2array(readtable("global_growth.csv", 'ReadVariableNames', false));
m = mats_prev;
mats_dyn = NaN(length(sample_countries),length(sample_countries),length(all_variants));
for i = 1:length(all_variants)
    s_i = S_global(i);
    mats_dyn(:,:,i) = m(:,:,i)*s_i;
end

writetable(array2table(nanmean(mats_dyn,3)),"mean_delay_S_global.csv");
writetable(array2table(nanmedian(mats_dyn,3)),"median_delay_S_global.csv");
writetable(array2table(nanstd(mats_dyn,[],3)),"std_delay_S_global.csv");
