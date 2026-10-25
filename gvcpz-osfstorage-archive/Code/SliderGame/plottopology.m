function plottopology(adjacency, tstring, figno, currstate, colors)

% PLOTTOPOLOGY Plot topology associated with symmetric adjaceny graph

figure(figno);
clf;
axes('FontSize', 18);
hold on;
axis off;
if nargin == 5
    nnodes = size(adjacency, 1);
    onnodes = currstate == 1;
    offnodes = currstate == -1;
    cmat = zeros(nnodes, 3);
    cmat(offnodes, :) = repmat(colors(1, :), [sum(offnodes), 1]);
    cmat(onnodes, :) = repmat(colors(2, :), [sum(onnodes), 1]);
    plot(graph(adjacency), 'LineWidth', 2.5, 'NodeFontSize', 18, 'EdgeColor', 'blue',...
        'NodeColor', cmat, 'MarkerSize', 10);
else
    plot(graph(adjacency), 'LineWidth', 2.5, 'NodeFontSize', 18, 'EdgeColor', 'blue')
end
title(tstring);
set(gca, 'units', 'normalized'); %Just making sure it's normalized
Tight = get(gca, 'TightInset');  %Gives you the border spacing between plot box and any axis labels
%[Left Bottom Right Top] spacing
NewPos = [Tight(1) Tight(2) 1-Tight(1)-Tight(3) 1-Tight(2)-Tight(4)]; %New plot position [X Y W H]
set(gca, 'Position', NewPos);
