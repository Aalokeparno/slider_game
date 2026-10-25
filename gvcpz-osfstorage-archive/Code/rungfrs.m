% Gradient Following with Random Switches (GFRS) Model 

% This file comes from the Open Science Framework (OSF) repository,
% https://osf.io/gvcpz/

% This model is discussed in Tabor, Smith, & Dankowicz (2024).  Escape
% from fraught states in a coordination game. To appear, February, 2024, 
% Royal Society Open Science.  See also https://osf.io/e5ztf

tic

addpath SliderGame

%%% Set parameters

% Model params
adjacency = [1 1 1 0 0 0 0;
    1 1 0 1 1 0 0;
    1 0 1 0 0 1 1;
    0 1 0 1 0 0 0;
    0 1 0 0 1 0 0;
    0 0 1 0 0 1 0;
    0 0 1 0 0 0 1];
delta = 0.1; 
meanstep = 20; % Called "mu" in Tabor et al. (2024)
tol = 0.05;
limvals = [-1, 1];
mtype = 'GFRS';

% Simulation parameters
maxtries = 10000;
timelimit = 300; 

% Analysis params
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

% Display params
printing = 0; % Set printing = 1 to save a jpeg version of each figure displayed in the directory, Data
%printing = 1;
reportinterval = 100;
figbase = 200;

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

% Initialize
maxp = 1.5*timelimit/delta; % for mem alloc

nplayers = size(adjacency, 1);
darray = zeros(nwarp+1, nplayers, nsought);
aarray = zeros(nwarp+1, nplayers, nsought);
frecord = zeros(nwarp+1, nsought);

successcount = 0;
dcount = 0;
trycount = 0;

fprintf('\nOf %d:  ', nsought);
while (dcount < nsought) && (trycount < maxtries)
    % Initialize
    trycount = trycount + 1;

    shist = zeros(nplayers, maxp);
    thist = zeros(1, maxp);
    shist(:, 1) = zeros(nplayers, 1); % Initialize to 0 state
    success = 0;
    dirvec = 2*rpick(2, [nplayers, 1]) - 3; % Randomly pick a direction (1 or -1) for every player
    posvec = delta*dirvec; % Go one microstep in the selected directions
    currtime = 0;
    nextind = rpick(nplayers);
    nexttime = -1; % Causes model to alter a direction on first update

    scount = 2;
    shist(:, scount) = posvec;

    % Run
    while ~success && (currtime < timelimit)
        scount = scount + 1;

        [posvec, dirvec, currtime, nexttime, nextind] = updatestatelocalgradrs(posvec, adjacency, dirvec, delta, currtime, nexttime, meanstep, nextind, limvals);

        thist(scount) = currtime;

        shist(:, scount) = posvec;

        success = nplayers - abs(sum(posvec)) < tol;
    end

    nsteps = scount;

    % Consolidate
    shist = shist(:, 1:nsteps);
    thist = thist(1:nsteps);
    totaltime = currtime;


    % Check winnning and amount of data
    if success && (scount >= 2)

        successcount = successcount + 1;

        % Time warp
        pwl = PWLF;
        inc = (totaltime-thist(1))/nwarp;
        pwl = pwl.buildf(shist', thist');
        newthist = thist(1):inc:totaltime;
        newthist = newthist';
        newshist = pwl.eval(newthist);

        % Compute fraughtness rate (this is an approximation because it applies
        % the fraghtness of each time window of the rasterized data to the whole
        % window, but with reasonably large nwarp, the value shouldn't be too far
        % off)
        nrstates = size(newshist, 1);
        fhist = zeros(1, nrstates);
        for scount = 1:nrstates
            fhist(scount) = fraught(newshist(scount, :)', adjacency);
        end
        pfraught = sum(fhist)/nrstates;

        % Selecåt relatively fraught cases

        if pfraught > fthresh
            dcount = report(dcount, reportinterval);

            % Flip so that winning side is 1
            if sum(newshist(end, :)) < 0
                newshist = -newshist;
            end

            % If need be, interchange the branches
            if mean(mean(newshist(:, rightbranch))) <...  % Reverse the branches if need be
                    mean(mean(newshist(:, leftbranch)))   % so Left Branch is always the one that reverses
                newshist = newshist(:, switchedorder);  % Switch player assignment of position raster
            end

            % Get agreements
            agreehist = zeros(size(newshist));
            for scount = 1:size(newshist, 1)
                agreehist(scount, :) = getagreelaopt(newshist(scount, :)', adjacency);
            end

            frecord(:, dcount) = fhist';
            darray(:, :, dcount) = newshist;
            aarray(:, :, dcount) = agreehist;
            % Track subgroups

            trycount = 0;
        end
    end
end
toc

fprintf('\nSuccess count = %d\n', successcount);

frecord = frecord(:, 1:dcount);
darray = darray(:, :, 1:dcount);
aarray = aarray(:, :, 1:dcount);

ncompleted = dcount;

dmean = mean(darray, 3);
amean = mean(aarray, 3);

%%% Plot by group
grouping = {1, [2, 4, 5], [3, 6, 7]};
gnames = {'Apex', 'Left Branch', 'Right Branch'};
gmeans = groupmeans(dmean, grouping);
ameans = groupmeans(amean, grouping);
% Reverse left branch
ameans(:, 2) = -ameans(:, 2);

mvals = mean(gmeans);

npoints = size(gmeans, 1);

% Mean trajectories
figure(figbase + 2);
setfig;
hold on;
xvals = (0:(npoints-1))/(npoints-1);
hh = zeros(3, 1);
for ccount = 1:3
    hh(ccount) = plot(xvals, gmeans(:, ccount), 'Color', colorlist(ccount, :));
end
for ccount = 1:3
    plot(xvals, ameans(:, ccount), 'Color', colorlist(ccount, :), 'LineWidth', 1);
end
xlabel('Normalized Time');
ylabel('Position');
title(mtype);
legend(hh, gnames);

%%% Printing
if printing % averaging analysis
    currdir = 'Data';
    if not(isfolder(currdir))
        mkdir(currdir)
    end
    figname = [currdir, '/', topname, '-Model-', mtype, '-nruns-', num2str(nruns), '-st-', num2str(meanstep), '-afig.jpg'];
    print('-djpeg', figname);
end
