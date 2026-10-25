%%% 2023-04-24

%%% This script selects the Confirmatory Lab Data analyses reported in 
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

currdataset = 'hcf';


%% The version in this section selects good Fall 2016 hcf runs
ntopologies = 11;  % See maketoplib in Dropbox/Projects/Oscillation/Analysis/analysis02InitialFall2016 for topology codes
outdat = [];
dcount = 0;
outtab = zeros(ntopologies, 2);
gcount = 0;  % Counts groups i.e. sets of participants from which data were successfully gathered
wavearray = {};
for fcount = 1:length(dirlist)
    fprintf('\n\n[[%d]]', fcount);
    
    currname = dirlist(fcount).name;
    if length(currname) > 3
        
        if isequal(currname(1:3), 'log')
            currfields = splitstring(currname, 'space');
            if length(currfields) > 3
                %%% Select good hcf runs:  these all occurred between Oct
                %%% 1 and Nov 16 (inclusive)
                firstparts = splitstring(currfields{1}, '_');
                if length(firstparts) == 2 |...   % The 2-field first parts are pre-version 09
                    (isequal(firstparts{2}, 'v09')... 
                        & ((isequal(currfields{2}, 'Sep')...
                                & ~isequal(currfields{4}, '162439')...     % cut-off
                                & ~isequal(currfields{4}, '170339')...     % cut-off
                                & ~isequal(currfields{4}, '153023'))...  % cut-off
                        | (isequal(currfields{2}, 'Oct')...
                                & ~isequal(currfields{3}, '31')... % Start of hcCTh
                                & ~isequal(currfields{4}, '154913')... % 15-person game
                                & ~isequal(currfields{4}, '155407')... % 15-person game
                                & ~isequal(currfields{4}, '164505')...  % 15-person game
                                & ~isequal(currfields{4}, '155839')...  % Exceptional hcCTh + error
                                & ~isequal(currfields{4}, '162500')...  % Exceptional hcCTh + error
                                & ~isequal(currfields{4}, '163226')...  % Exceptional hcCTh
                                & ~isequal(currfields{4}, '171748')...  % Exceptional hcCTh + error
                                & ~isequal(currfields{4}, '160321')...  % Exceptional asterisk
                                & ~isequal(currfields{4}, '170411')...  % Exceptional asterisk
                                & ~isequal(currfields{4}, '170439')...  % Exceptional asterisk
                                & ~isequal(currfields{4}, '173032')...  % Exceptional asterisk
                                & ~isequal(currfields{4}, '180403'))...  % Exceptional asterisk
                                )...
                     )
  
                            datafilename = [dirname, '/', dirlist(fcount).name];
                            gcount = gcount + 1;
                            
                            wavearray{gcount} = loaddata(datafilename);  % Run
                            %without this to count groups, then run with it to load
                            %data.
                end
            end
        end
    end
end
        
ngroups = gcount

        
