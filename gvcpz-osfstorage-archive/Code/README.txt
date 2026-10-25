This repository contains Matlab code for running simulations of the Slider Game
as reported in

    Tabor, W., Smith, G., & Dankowicz, H. (2024).  Escape from fraught states
    in the Slider Game. Royal Society Open Science. A preprint is 
    available at https://osf.io/preprints/osf/e5ztf

Instructions:

    Download all files, including the directory, SliderGame.

    Set Matlab's working directory to the directory where you downloaded the
    files.

    From the command window, 

    type

        runapem;

    to run the Agreement Position Emigration (APEM) model. 
    (This should produce a figure showing the Hierarchy Topology on 7 players
    [like Figure 1a from the paper] and a figure showing the Averaging Analysis
    of the APEM model [like Figure 6 of the paper].

    Type

        runrw

    to run the Random Walk (RW) model. 
    (This should produce a figure showing the Hierarchy Topology on 7 players
    [like Figure 1a from the paper] and a figure showing the Averaging Analysis
    of the RW model [like Figure 4 in the supplementary materials of the paper].

    Type

        rungfrs

    to run the Gradient Following with Random Switches (GFRS) model. 
    (This should produce a figure showing the Hierarchy Topology on 7 players
    [like Figure 1a from the paper] and a figure showing the Averaging Analysis
    of the GFRS model [like Figure 5 in the supplementary materials of the paper].


    

