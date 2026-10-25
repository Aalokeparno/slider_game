% Agreement Position Emigration (APEM) Model 

% This file comes from the Open Science Framework (OSF) repository,
% https://osf.io/gvcpz/

% This model is discussed in Tabor, Smith, & Dankowicz (2024).  Escape
% from fraught states in a coordination game. To appear, February, 2024, 
% Royal Society Open Science.  See also https://osf.io/e5ztf

tic

addpath SliderGame

%%% Set parameters

% Model parameters
adjacency = [1 1 1 0 0 0 0;
    1 1 0 1 1 0 0;
    1 0 1 0 0 1 1;
    0 1 0 1 0 0 0;
    0 1 0 0 1 0 0;
    0 0 1 0 0 1 0;
    0 0 1 0 0 0 1];
topname = 'Hierarchy-7';
limvals = [-1, 1];
atheta = 0.5; 
stheta = 0.5;
meanstep = 0.1; % Called "mu" in Tabor et al. (2024)
mtype = 'APEM';

% Simulation parameters
timelimit = 300;
maxp = 10000;

% Analysis params
leftbranch = [2, 4, 5];
rightbranch = [3, 6, 7];
apex = 1;
grouping = {apex, leftbranch, rightbranch};
gnames = {'Apex', 'Left Branch', 'Right Branch'};
switchedorder = [1, 3, 2, 6, 7, 4, 5];
leftcolor = [0.5, 0.5, 0];
rightcolor = [0.5, 0, 0.5];
apexcolor = [0.8, 0, 0];
fullagreecolor = [160, 82, 45]/256;  % Sienna
colorlist = [apexcolor; leftcolor; rightcolor];
nwarp = 1000;
nsought = 1000; % Number of runs to seek satisfying percent fraught (pfraught) > fthresh
maxtries = 10000;
fcutoff = 0.1; % Fraughtness portion cutoff (only runs with at least this 
               % portion of fraughtness will be included in the analysis)

% Display parameters
reportinterval = 100;
figbase = 300;
%printing = 0; % Set printing = 1 to save a jpeg version of each figure displayed in the directory, Data
printing = 1;

%%% Plot the network topology
topfigno = figbase + 1;
tstringlocal = [topname, ' topology'];
plottopology(adjacency, tstringlocal, topfigno);
topfigname = [topname, '-topology.jpg'];
if printing
    currdir = 'Data';
    if not(isfolder(currdir))
        mkdir(currdir)
    end
    figname = [currdir, '/', topfigname];
    print('-djpeg', figname);
end

%%% Run the games

nplayers = size(adjacency, 1);
darray = zeros(nwarp+1, nplayers, nsought);
aarray = zeros(nwarp+1, nplayers, nsought);
frecord = zeros(nwarp+1, nsought);

successcount = 0;
dcount = 0;
trycount = 0;

tic
fprintf('\nOf %d:  ', nsought);
while (dcount < nsought) && (trycount < maxtries)
    % Initialize
    trycount = trycount + 1;

    shist = zeros(nplayers, maxp);
    lhist = zeros(1, maxp);
    thist = zeros(1, maxp);
    shist(:, 1) = zeros(nplayers, 1); % Initialize to 0 state
    lhist(1) = 0;
    success = 0;
    dirvec = 2*rpick(2, [nplayers, 1]) - 3; % Randomly pick a direction (1 or -1) for every player
    posvec = meanstep*dirvec; % Go one meanstep in the selected directions
    currtime = meanstep;

    scount = 2;
    shist(:, scount) = posvec;
    lhist(scount) = meanstep;

    % Run
    while ~success && (currtime < timelimit)
        scount = scount + 1;

        [posvec, dirvec, nextind, nextlat] = updatestatecapem(posvec, dirvec, adjacency, meanstep, atheta, stheta, limvals);

        lhist(scount) = nextlat;
        currtime = currtime + nextlat;
        thist(scount) = currtime;

        shist(:, scount) = posvec;

        success = abs(sum(posvec)) == nplayers;
    end
    nsteps = scount;

    % Consolidate
    shist = shist(:, 1:nsteps);
    lhist = lhist(1:nsteps); % Sequence of latencies
    thist = thist(1:nsteps);
    totaltime = currtime;

    % Check winning

    if success

        successcount = successcount + 1;

        % Time warp
        pwl = PWLF;
        inc = (totaltime-thist(1))/nwarp;
        pwl = pwl.buildf(shist', thist');
        newthist = thist(1):inc:totaltime;
        newthist = newthist';
        newshist = pwl.eval(newthist);

        % Get fraughtness rate
        nrstates = size(newshist, 1);
        fhist = zeros(1, nrstates);
        for scount = 1:nrstates
            fhist(scount) = fraught(newshist(scount, :)', adjacency);
        end
        pfraught = sum(fhist)/nrstates;

        % Selecåt relatively fraught cases

        if pfraught > fcutoff
            dcount = report(dcount, reportinterval);

            % Flip so that winning side is 1
            if sum(newshist(end, :)) < 0
                newshist = -newshist;
            end

            % If need be, interchange the branches
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

            % Store data
            frecord(:, dcount) = fhist';
            darray(:, :, dcount) = newshist;
            aarray(:, :, dcount) = agreehist;

            trycount = 0;
        end
    end
end
toc

fprintf('\nSuccess count = %d\n', successcount);

frecord = frecord(:, 1:dcount);
darray = darray(:, :, 1:dcount);
aarray = aarray(:, :, 1:dcount);

nruns = dcount;

dmean = mean(darray, 3);
amean = mean(aarray, 3);

%%% Plot by group

figure(figbase + 2);
setfig;
hold on;

gmeans = groupmeans(dmean, grouping);
ameans = groupmeans(amean, grouping);
% Reverse left branch
if ~isequal(topname(1:4), 'full')
    ameans(:, 2) = -ameans(:, 2);
end
tvals = (0:nwarp)/(nwarp+1);

%%% Find change pts and plot adverse
%%% NOTE:  In order for this section to run, Matlab's Signal Processing Toolbox must be
%%% installed
% % Find change points
% leftbranchmean = gmeans(:, 2);
% ipt = findchangepts(leftbranchmean', 'MaxNumChanges', 3, 'Statistic', 'linear');
% % Plot adverse
% fcolor = [0.8, 0.8, 0.8];
% leftind = tvals(ipt(2));
% rightind = tvals(ipt(3));
% xvals = [leftind, rightind, rightind, leftind];
% %yvals = [limvals(1), limvals(1), limvals(2), limvals(2)];
% yvals = [-1.5, -1.5, 1.5, 1.5]; % Overkill
% fill(xvals, yvals, fcolor);

% Mean trajectories
hh = zeros(3, 1);
for ccount = 1:3
    hh(ccount) = plot(tvals, gmeans(:, ccount), 'Color', colorlist(ccount, :));
end
if ~isequal(topname(1:4), 'full')
    for ccount = 1:3
        plot(tvals, ameans(:, ccount), 'Color', colorlist(ccount, :), 'LineWidth', 1);
    end
else
    plot(tvals, ameans(:, 1), 'Color', fullagreecolor, 'LineWidth', 1);
end

xlabel('Normalized Time');
ylabel('Position');
title(mtype);
legend(hh, gnames);
ylim(limvals);

%%% Printing
if printing % averaging analysis
    currdir = 'Data';
    if not(isfolder(currdir))
        mkdir(currdir)
    end
    figname = [currdir, '/', topname, '-Model-', mtype, '-nruns-', num2str(nruns), '-fcutoff-', num2str(fcutoff), '-meanstep-', num2str(round(meanstep, 4)), '-afig.jpg'];
    print('-djpeg', figname);
end

toc

