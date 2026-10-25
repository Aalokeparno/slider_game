function yysum = groupmeans(yy, grouping)

% GROUPMEANS cell array grouping specifieds subsets of the columns of yy
% return the mean of the columns of each group

yysum = zeros(size(yy, 1), length(grouping));

for gcount = 1:length(grouping)
    yysum(:, gcount) = mean(yy(:, grouping{gcount}), 2);
end