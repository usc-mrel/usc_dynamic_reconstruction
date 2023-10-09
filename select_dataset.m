% Select a dataset to use.
% Pre-selected datasets one good and one bad each.
% -------------------------------------------------------------------------
% Summary of the datasets are here:
% 
% Cardiac:
% 
% Speech:
% 
% -------------------------------------------------------------------------

switch area
    % ---------------------------------------------------------------------
    case 'cardiac'
        switch which_file
            % -------------------------------------------------------------
            % "Good" example
            case 0
                path = "/server/sdata/eyagiz/mri_data/disc/heart/vol0376_20220620/raw_hawk/";
                name = "usc_disc_yt_2022_06_20_144717_multi_slice_golden_angle_spiral_ssfp_slice_9_fov_240_n17_slice_06.mat";
            % -------------------------------------------------------------
            % "Bad" example
            case 1
        end
    % ---------------------------------------------------------------------
    case 'speech'
        switch which_file
            % -------------------------------------------------------------
            % "Good" example
            case 0 
                path = "/server/home/pkumar/mri_data/disc/speech/vol0634_20230601/raw_hawk/";
                name = "usc_disc_20230601_172615_pk_speech_rt_ssfp_fov24_res24_n13_vieworder_bitr.mat";
            % -------------------------------------------------------------
            % "Bad" example
            case 1
        end
    % ---------------------------------------------------------------------
end