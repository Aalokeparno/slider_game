function mm = crop(dd, lb, ub)

% CROP bring values of dd within the bounds, lb, ub

mm = min(dd, ub);

mm = max(mm, lb);