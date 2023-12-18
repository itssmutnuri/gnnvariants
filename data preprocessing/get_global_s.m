%There are negative numbers, but that is because I did not set the s value of the first variant. But basically, we can take the smallest one (B.1.2) and its value to all others
full_A = []; full_b = [];
for cc = 1:size(S_rel, 1)
    xx = squeeze(S_rel(cc, :, :));
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
    full_A = [full_A; A]; full_b = [full_b; b];
end
%%
val_idx = find(any(full_A>0, 1));
A1 = full_A(:, val_idx);
s = A1\full_b; s = s - min(s);
% Show results
%%
% s = full_A\full_b; s = s - min(s);
%%
res = [all_variants(val_idx) num2cell(s)];
if isempty(res)
    s = full_A\full_b; s = s - min(s);
    res = [all_variants num2cell(s)];
end
T  = cell2table(res,'VariableNames',{'variant', 'S'});
% writetable(T,'growth_rates_clustered_new.csv')