%%% 2023-04-24

%%% This script selects the Exploratory Lab Data analyses reported in 
%%% Tabor, W., Smith, G., & Dankowicz, H. (2023). Escaping fraught states 
%%% in a coordination game. psyarxiv.org.

%%% After running this script in Matlab or Octave, data will be stored in an array 
%%% called wavearray, each entry of which has the data from one group's runs.

%%% Note:  the specific data selected are selected by date of running in
%%% the if statement below starting around line 43

dirname = 'Data-2016-12-09'; % Set dirname to the name of the folder containing
                             % the data log files

dirlist = dir(dirname);

fprintf('\n\nOf:  %d', length(dirlist));

ngroups = 17;

currdataset = 'hcCTh';

%% The version in this section selects good Fall 2016 hcCTh runs
ntopologies = 11;  % See maketoplib in Dropbox/Projects/Oscillation/Analysis/analysis02InitialFall2016 for topology codes
outdat = [];
dcount = 0;
outtab = zeros(ntopologies, 2);
gcount = 0;  % Counts groups i.e. sets of participants from which data were successfully gathered
wavearray = cell(1, ngroups);
for fcount = 1:length(dirlist)
    fprintf('\n\n[[%d]]', fcount);
    
    currname = dirlist(fcount).name;
    if length(currname) > 3
        
        if isequal(currname(1:3), 'log')
            currfields = splitstring(currname, 'space');
            if length(currfields) > 3
                %%% Select good hcCTh runs:  these all occurred between Oct
                %%% 31 and Nov 16 (inclusive)
                %%% Eliminate two files which ran hcf by mistake:  161103 and 161208
                if (isequal(currfields{2}, 'Nov') & ~(str2num(currfields{3}) > 16)...
                        & ~isequal(currfields{4}, '161103') & ~isequal(currfields{4}, '161208'))...  % These ran hcf by mistake
                        | (isequal(currfields{2}, 'Oct') & isequal(currfields{3}, '31'))
                    
                    
                    datafilename = [dirname, '/', dirlist(fcount).name];
                    gcount = gcount + 1;
                    
                    wavearray{gcount} = loaddata(datafilename);  % Run
                    %without this to count groups, then run with it to load
                    %data.
                end;
            end;
        end;
    end;
end;

ngroups = gcount;



