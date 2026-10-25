This file contains instructions for accessing the data that were used in the 
analyses reporte in 

	Tabor, W., Smith, G., & Dankowicz, H. (2023). Escaping fraught states 
	in a coordination game. psyarxiv.org.
	
Instructions:

1.  Download the two files,

	Data-1016-12-09-tabor-smith-dankowicz-psyarxiv-2023-escaping-fraught-states-cc.zip
	Data-2018-05-31-tabor-smith-dankowicz-psyarxiv-2023-escaping-fraught-states-cc.zip
	
	and uncompress them.
	
2.  Download one of the Matlab/Octave scripts,

	selectdatasetEL.m
	selectdatasetCL.m
	selectdatasetCC.m
	
	Each of these scripts compiles the data for one of the human participant runs
	reported in the paper above:
	
		EL = Exploratory Lab Data Set
		CL = Confirmatory Lab Data Set
		CC = Confirmatory Cloud Data Set
		
	Before running the script, you'll need to edit it and set the variable dirname
	(line 13) to the value of the directory name where you've stored the data.
		
3.  Run the script you selected in Step 2 in Matlab or Octave.  The data
	will then be stored in a variable called wavearray, a two dimensional
	cell array.  You can inspect the data for one game run by typing
	wavarray{groupid}{gameid} (e.g. wavearray{1}{1}) on the command line.