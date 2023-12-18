clear
load objs21N_rel_clustered


S = S_rel;
s0 = 1;
SA = S - s0;
% SA(isnan(SA)) = 0;
%%
var_prev_S = readtable('splined_prevT21i_clusteredF_GTA.csv');%readtable('splined_prevTF21i.csv');
countries = table2array(readtable("countries_clustered.csv", 'ReadVariableNames', false));
var_prev = var_prev_S;

% Assign the sample countries
sample_countries = countries; %["USA", "United Kingdom", "China", "Australia", "India"] #S

% Get all unique variants
all_variants = readtable("all_vars21_clustered_new.csv").all_variants;
%%
%197x426x426 should be squeezed into (197x426)x426  (eqns
% S = reshape(S,[],426);
A = zeros(1,size(S_rel,2));
b = [];
count = 0;
for i=1:size(S_rel,1)
%     eqn = s(1);
    for j=1:size(S_rel,2)
        if ~any(~isnan(S_rel(i,j,:)))%isnan(S_rel(i,j,:)) %If no data on this, skip
            continue
        end
        rel_mat = squeeze(S_rel(i,j,:));        
        for k = 1:size(S_rel,3)
            if isnan(rel_mat(k))
                continue
            end
            count = count + 1;
            b(count) = S_rel(i,j,k);
            A(count,j) = 1;
        end
        %eqns(count) = (s(j) - s0 -SA(i,j)).^2;
        
    end
end
b= b';
lb = zeros(1, size(S_rel,2)) + 0.001;
x = lsqlin(A,b,[],[],[],[],lb,[]);

hist(x); 
x(313)
x = array2table(x, 'VariableNames', {'global_S'});

% col_names = {'variant', 'global_S'};

x = addvars(x,all_variants, 'Before',1);
% Write table to CSV file
writetable(x, 'global_growth_N.csv');
%%
s = sym('s', [1 size(S,2)]);
N = sum(~isnan(SA(:)));
eqns = sym(zeros(N, 1));
count = 0;
for i=1:size(SA,1)
%     eqn = s(1);
    for j=1:size(SA,2)
        if isnan(SA(i,j))
            continue
        end
        count = count + 1;
        eqns(count) = (s(j) - s0 -SA(i,j)).^2;
        
    end
end

te = [sum(eqns)==0];
%or slove() solve(te,s)
obj = @(x) sum(double(subs(eqns,s,x)));
%%

% N = 400; % number of variants
% M = 197; % number of countries
% % generate random growth advantage data for each country and variant
% % grow_advantages = randn(N, N, M);
% % initialize var_mat matrix with zeros
% var_mat = zeros(N, M);
% define the objective function to be minimized
% fun = @(s) sum((S*(s')-S(:,1)).^2);
fun = @(s) sum((s'-s1-SA(:)).^2);


% fun = @(s) sum((s'- S').^2);
%%
% set initial guess for the solver to start from
x0 = rand(1,size(S,2)); %rand(1, N*M);
% set lower and upper bounds for s1,s2,...,sM to be between 0 and 1
lb = zeros(1, size(S,2));
% ub = ones(1, N*M);
% set options for the solver
options = optimoptions('fmincon','Display','iter','Algorithm','sqp');
% solve the optimization problem using fmincon
x = fmincon(obj,x0,[],[],[],[],lb,[],[],options);
% reshape the solution to a NxM matrix
% var_mat = reshape(x, size(S,2),size(S,2));
%%
xx = squeeze(S_rel(34, :, :));
A = zeros(sum(~isnan(xx(:))), size(xx, 1));
b = zeros(sum(~isnan(xx(:))), 1);
j  = 1;
for i=1:size(xx, 1)
    base_list = find(~isnan(xx(i, :)));
    for k=1:length(base_list)
        A(j, i) = 1;
        A(j, base_list(k)) = -1;
        b(j) = xx(i, base_list(k));
        j = j+1;
    end
end
%%
val_idx = find(any(A>0, 1));
A1 = A(:, val_idx);
s = A1\b
% Show results
[all_variants(val_idx) num2cell(s)]