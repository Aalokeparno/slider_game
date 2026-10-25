%%% 2023-04-24

%%% This script selects the Confirmatory Cloud Data analyses reported in 
%%% Tabor, W., Smith, G., & Dankowicz, H. (2023). Escaping fraught states 
%%% in a coordination game. psyarxiv.org.

%%% After running this script in Matlab or Octave, data will be stored in an array 
%%% called wavearray, each entry of which has the data from one group's runs.

%%% Note:  the specific data selected are selected by date of running in
%%% the if statement below starting around line 43

dirname = '2018-05-31 RunsScrubbed'; % Set dirname to the name of the folder containing
                             % the data log files

nplayers = 7;

dirlist = dir(dirname);

fprintf('\n\nOf:  %d', length(dirlist));

currdataset = 'summer2018benchmark';

ntopologies = 14;  % See maketoplib
[alib, typeconv, simpletypes] = maketoplib(nplayers);
outdat = [];
dcount = 0;
outtab = zeros(ntopologies, 2);
gcount = 0;  % Counts groups i.e. sets of participants from which data were successfully gathered
wavearray = {};
topcounts = [];
toplist = {};
for fcount = 1:length(dirlist)
    %for fcount = 1:7
    fprintf('\n\n[[%d]]', fcount);
    
    currname = dirlist(fcount).name;
    
    if length(currname) >= 4
        if currname((end-3):end) == '.txt'
            
            datafilename = [dirname, '/', dirlist(fcount).name];
            gcount = gcount + 1;
            
            wavearray{gcount} = loadsessionDweb(datafilename, nplayers, alib, typeconv);
           
               if isequal(datafilename, 'Log_2018-05-31_135852.txt')
                wavearray{gcount} = {wavearray{gcount}{[1:6, 8:9]}};  % Something amiss with 6th/7th wave
            end;
            
            if isequal(datafilename, 'Log_2018-05-31_140824.txt')
                wavearray{gcount} = {wavearray{gcount}{[1:6, 8:9]}};  % Something amiss with 6th/7th wave
            end;
           
            if isequal(datafilename, 'Log_2018-05-31_145138.txt')
                wavearray{gcount} = {wavearray{gcount}{1:8}};  % Last wave of this session was degenerate
            end;
            
            if isequal(datafilename, 'Log_2018-05-31_160337.txt')
                wavearray{gcount} = {wavearray{gcount}{[1, 3:6]}};  % Something amiss with 2nd/3rd wave
                                                                    % (And
                                                                    % last 3 are missing)
            end;
            
            
            if isequal(datafilename, 'Log_2018-05-31_163720.txt')  % group 16
                wavearray{gcount} = {wavearray{gcount}{[1:6, 8:9]}};  % Something amiss with 6th/7th wave
            end;
            
            if isequal(datafilename, 'Log_2018-05-31_182335.txt')  % group 24
                wavearray{gcount} = {wavearray{gcount}{1:6}};  % Something amiss after 6th
            end;         
            
            for wcount = 1:length(wavearray{gcount})
                if wcount <= 9  % This is due to a strange occurrence in session Log_2018-05-31_145138:
                    % 10 waves were started and the last one
                    % seems degenerate
                    currwave = wavearray{gcount}{wcount};
                    if ~currwave.robotspresent % Only take non robot waves
                        currtop = getindcell(currwave.topology, toplist);
                        if currtop == 0
                            toplist = [toplist, currwave.topology];
                            topcounts = [topcounts, 1];
                        else
                            topcounts(currtop) = topcounts(currtop) + 1;
                        end;
                    end;
                end;
            end;
        end;
    end;
end;

ngroups = gcount;



