% Random Walk Model 

% This file comes from the Open Science Framework (OSF) repository,
% https://osf.io/gvcpz/

% This model is discussed in Tabor, Smith, & Dankowicz (2024).  Escape
% from fraught states in a coordination game. To appear, February, 2024, 
% Royal Society Open Science.  See also https://osf.io/e5ztf

tic

addpath SliderGame

%%% Set parameters

% Model parameters
% Case: Hierarchy
topname = 'hierarchy-7';
adjacency = [1 1 1 0 0 0 0; % hierarchy
    1 1 0 1 1 0 0;
    1 0 1 0 0 1 1;
    0 1 0 1 0 0 0;
    0 1 0 0 1 0 0;
    0 0 1 0 0 1 0;
    0 0 1 0 0 0 1];
delta = 2; % Constant flipping
mtype = 'RW';

% Simulation parameters
timelimit = 150;

% Analysis parameters
fthresh = 0.1;
leftbranch = [2, 4, 5];
rightbranch = [3, 6, 7];
apex = 1;
switchedorder = [1, 3, 2, 6, 7, 4, 5];
leftcolor = [0.5, 0.5, 0];
rightcolor = [0.5, 0, 0.5];
apexcolor = [0.8, 0, 0];
colorlist = [apexcolor; leftcolor; rightcolor];
nwarp = 1000;
nsought = 1000; % Number of runs to seek satisfying percent fraught (pfraught) > fthresh

% Display parameters
figbase = 100;
reportinterval = 100;
printing = 0; % Set printing = 1 to save a jpeg version of each figure displayed in the directory, Data
%printing = 1;

%%% Plot the topology
topfigno = figbase + 1;
tstringlocal = [topname, ' topology'];
plottopology(adjacency, tstringlocal, topfigno);

% If printing, print the topology
topfigname = [currdir, '/', topname, '-topology.jpg'];
if printing
    currdir = 'Data';
    if not(isfolder(currdir))
        mkdir(currdir)
    end
    figname = [currdir, '/', topfigname];
    print('-djpeg', topfigname);
end

%%% Run the games

% Initialize
nplayers = size(adjacency, 1);
darray = zeros(nwarp, nplayers, nsought);
aarray = zeros(nwarp, nplayers, nsought);
dcount = 0;
successcount = 0;

fprintf('\nOf %d:  ', nsought);
while dcount < nsought

    % Initialize
    ndim = size(adjacency, 1);
    currstate = zeros(ndim, 1);
    shist = zeros(ndim, timelimit);
    shisst(:, 1) = zeros(ndim, 1); % Initialize to 0 state
    success = 0;
    scount = 1;

    %%% Run

    while ~success && (scount < timelimit)
        scount = scount + 1;
        ds = rpick(3, [ndim, 1]) - 2;
        currstate = crop(currstate + delta*ds, -1, 1);
        shist(:, scount) = currstate;

        success = abs(sum(currstate)) == ndim;
    end
    nsteps = scount;

    % Consolidate, transpose
    shist = shist(:, 1:nsteps);
    shist = shist';


    % Check winning
    if success

        successcount = successcount + 1;
        fvals = zeros(1, nsteps);
        for scount = 1:nsteps
            fvals(scount) = fraught(shist(scount, :)', adjacency);
        end
        pfraught = sum(fvals == 1)/nsteps;

        % Select relatively fraught cases

        if pfraught >= fthresh
            dcount = report(dcount, reportinterval);

            % Flip so that winning side is 1
            if sum(shist(end, :)) < 0
                shist = -shist;
            end

            % Time warp
            pwl = PWLF;
            rawlength = size(shist, 1);
            inc = rawlength/nwarp;
            pwl = pwl.buildf(shist);
            newshist = pwl.eval(inc:inc:rawlength);

            % Flip so that the Right Branch is 1-biased
            if mean(mean(newshist(:, rightbranch))) <...  % Reverse the branches if need be
                    mean(mean(newshist(:, leftbranch)))   % so Left Branch is always the one that reverses
                newshist = newshist(:, switchedorder);  % Switch player assignment of position raster
                %curragrnewshist = curragrnewshist(:, switchedorder);  % Correspondingly, switch player assignment of agreement raster
            end

            % Get agreements
            agreehist = zeros(size(newshist));
            for scount = 1:size(newshist, 1)
                agreehist(scount, :) = getagreelaopt(newshist(scount, :)', adjacency);
            end

            darray(:, :, dcount) = newshist;
            aarray(:, :, dcount) = agreehist;

            % Track subgroups

        end
    end
end

fprintf('\nSuccess count = %d\n', successcount);

nruns = dcount;

dmean = mean(darray, 3);
amean = mean(aarray, 3);

%%% Check fraughtness for aligned data

tic
fcount = 0;
findex = zeros(nwarp, 1);
for scount = 1:nwarp
    if fraught(dmean(scount, :)', adjacency)
        fcount = fcount + 1;
        findex(scount) = 1;
    end
end
mfraughtness = fcount/nwarp;
findex = logical(findex);

%%% Plot by group

grouping = {1, [2, 4, 5], [3, 6, 7]};
gnames = {'Apex', 'Left Branch', 'Right Branch'};
gmeans = groupmeans(dmean, grouping);
ameans = groupmeans(amean, grouping);
% Reverse left branch
ameans(:, 2) = -ameans(:, 2);

%%% Plot by group
figure(figbase + 2);
hold off;
setfig;
hold on;
hh = zeros(3, 1);
for ccount = 1:3
    hh(ccount) = plot(gmeans(:, ccount), 'Color', colorlist(ccount, :));
end
for ccount = 1:3
    plot(ameans(:, ccount), 'Color', colorlist(ccount, :), 'LineWidth', 1);
end
xlabel('Normalized Time');
ylabel('Position');
title(mtype);
legend(hh, gnames);

if printing % averaging analysis
    currdir = 'Data';
    if not(isfolder(currdir))
        mkdir(currdir)
    end
    figname = [currdir, '/', topname, '-Model-', mtype, '-nruns-', num2str(nruns), '-fthresh-', num2str(fthresh), '-afig.jpg'];
    print('-djpeg', figname);
end

toc