function [Kconc] = Kconv(Ktr, fs)
%% MatLab script to convert the K-electrode measures into an (approximate) [K]
% It assumes sampling time interval, r = 0.001, and utilises the basic
% formula provided by Juha Voipio as follows:
% [K] = 3.5 * 10^((tr-offset)/S)
% where tr is the measurements taken using the K electrode, and S is the
% calibration scale, which is assumed to be 55 - if not, then change in
% line 15

%% inputs for original coding

S = 55;

% Shortened traces
Kf = Ktr;  

%% Notch filter - apply to ionophore channel
bsFilt = designfilt('bandstopiir', 'FilterOrder', 20, ...
    'HalfPowerFrequency1', 48, 'HalfPowerFrequency2', 52, ...
    'SampleRate', fs);
Kf = filtfilt(bsFilt,Kf);


%%
Kconc = 3.5 * 10.^(Kf/S);
