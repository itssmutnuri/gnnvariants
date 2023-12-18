function mean_tau = trivial_m(retro_table,mats_prev,all_variants)
    vars = unique(retro_table.pangoLineage);    
    mean_tau = nanmean(mats_prev(:,:,ismember(vars,all_variants)),3);
end