function [powerTable,values,categories] = BandPowerNormalize(signal,sampRate,start,finish,ambient,varargin)
%SIGNALBANDANALYSIS Inputs: signal (whole signal), sampRate, start, finish, ambient (Boolean), amb_start
% (optional), amb_finish (optional)
% Optional inputs
if nargin < 6
        nBlank = 0; 
 else
    amb_start = varargin{1};
    amb_finish = varargin{2};
    Desc = varargin{3};
 end


% Once we have user input, add 1 sec to the outer bounds for the selected area
start_ind = find(signal(:,1) > start-1, 1); % 1sec 
finish_ind = find(signal(:,1) > finish+1, 1); % 1 second
interesting_signal = signal(start_ind:finish_ind,:);

if ambient == false
    % ambient signal
    start_ind_amb = find(signal(:,1) > amb_start-1,1);
    finish_ind_amb = find(signal(:,1) > amb_finish+1, 1);
    ambient_signal = signal(start_ind_amb:finish_ind_amb,:);

    %Hanning window to select the interesting signal
    window = custom_hanning(sampRate,length(interesting_signal(:,1)));
    interesting_signal(:,2) = interesting_signal(:,2) .* window;

    %Taper to baseline
    baseline = mean(ambient_signal(:,2));
    baselineIndices = find(window<1);
    interesting_signal(baselineIndices,2) = interesting_signal(baselineIndices,2) + baseline*(1-window(baselineIndices));
    %Ambient Power
    [~, ambPowers, ~] = AmbPower(signal,sampRate,amb_start,amb_finish,true);
end

% FFT the interesting signal
INTERESTING_SIGNAL = fft(interesting_signal(:,2));
% INTERESTING_SIGNAL = INTERESTING_SIGNAL * (1 / length(INTERESTING_SIGNAL));
% Compute the FFT

% Calculate the frequency axis
N = length(INTERESTING_SIGNAL);
frequencies = sampRate*(0:(N/2))/N;
frequencies = transpose(frequencies);


% Calculate the magnitude of the FFT result
magnitude = abs(INTERESTING_SIGNAL(1:(round(N/2))));
% match lengths
if length(frequencies) > length(magnitude)
    matchLen = length(magnitude);
    frequencies = frequencies(1:matchLen);
end
if length(frequencies) < length(magnitude)
    matchLen = length(frequencies);
    magnitude = frequencies(1:matchLen);
end

ind_60 = filter_mask(frequencies, 60);
ind_120 = filter_mask(frequencies, 120);
ind_180 = filter_mask(frequencies, 180);
ind_240 = filter_mask(frequencies, 240);
ind_300 = filter_mask(frequencies, 300);
ind_360 = filter_mask(frequencies, 360);
ind_420 = filter_mask(frequencies, 420);
ind_480 = filter_mask(frequencies, 480);
mask = ind_60+ind_120+ind_180+ind_240+ind_300+ind_360+ind_420+ind_480;
magnitude_filtered = magnitude;
indexes = find(mask==1);
magnitude_filtered(indexes) = NaN;

%Time domain calculation:
%totPowerT = sum(interesting_signal(:,2).^2) / length(interesting_signal(:,2));
%Freq domain calculation:
dt = 1 / sampRate;
T = dt * length(interesting_signal(:,2));
%[~, totPowerF] = msvF(INTERESTING_SIGNAL, T, dt);


% Delta band filtered data

% Define the bandpass filter in the frequency domain
deltaLow_freq = 1;           % Low cutoff frequency (Hz)
deltaHigh_freq = 4;          % High cutoff frequency (Hz)

% Create the frequency-domain filter mask
deltaFilter_mask = (frequencies >= deltaLow_freq) & (frequencies <= deltaHigh_freq);

% Apply the filter to the FFT data
fft_filtered_delta = magnitude(1:end) .* deltaFilter_mask;

%Correction for bins not being exactly on cutoffs
indices = find(deltaFilter_mask == 1);
firstIndex = min(indices);
lastIndex = max(indices);
lowBinPer = (frequencies(firstIndex)-deltaLow_freq)/(frequencies(firstIndex)-frequencies(firstIndex-1));
lowCorPower = lowBinPer*(frequencies(firstIndex)-frequencies(firstIndex-1))*(.5*abs(magnitude(firstIndex)+magnitude(firstIndex-1)));
highBinPer = (deltaHigh_freq-frequencies(lastIndex))/(frequencies(lastIndex+1)-frequencies(lastIndex));
highCorPower = highBinPer*(frequencies(lastIndex)-frequencies(lastIndex-1))*(.5*abs(magnitude(lastIndex)+magnitude(lastIndex+1)));

[~, deltaPower] = msvF(fft_filtered_delta, T, dt);
deltaPower = deltaPower + highCorPower +lowCorPower;

% Theta band filtered data

% Define the bandpass filter in the frequency domain
thetaLow_freq = 4;           % Low cutoff frequency (Hz)
thetaHigh_freq = 8;          % High cutoff frequency (Hz)

% Create the frequency-domain filter mask
thetaFilter_mask = (frequencies >= thetaLow_freq) & (frequencies <= thetaHigh_freq);

% Apply the filter to the FFT data
fft_filtered_theta = magnitude(1:end) .* thetaFilter_mask;

%Correction for bins not being exactly on cutoffs
indices = find(thetaFilter_mask == 1);
firstIndex = min(indices);
lastIndex = max(indices);
lowBinPer = (frequencies(firstIndex)-thetaLow_freq)/(frequencies(firstIndex)-frequencies(firstIndex-1));
lowCorPower = lowBinPer*(frequencies(firstIndex)-frequencies(firstIndex-1))*(.5*abs(magnitude(firstIndex)+magnitude(firstIndex-1)));
highBinPer = (thetaHigh_freq-frequencies(lastIndex))/(frequencies(lastIndex+1)-frequencies(lastIndex));
highCorPower = highBinPer*(frequencies(lastIndex)-frequencies(lastIndex-1))*(.5*abs(magnitude(lastIndex)+magnitude(lastIndex+1)));

[~, thetaPower] = msvF(fft_filtered_theta, T, dt);
thetaPower = thetaPower + highCorPower +lowCorPower;

% Alpha band filtered data

% Define the bandpass filter in the frequency domain
alphaLow_freq = 8;           % Low cutoff frequency (Hz)
alphaHigh_freq = 12;          % High cutoff frequency (Hz)

% Create the frequency-domain filter mask
alphaFilter_mask = (frequencies >= alphaLow_freq) & (frequencies <= alphaHigh_freq);

% Apply the filter to the FFT data
fft_filtered_alpha = magnitude(1:end) .* alphaFilter_mask;

%Correction for bins not being exactly on cutoffs
indices = find(alphaFilter_mask == 1);
firstIndex = min(indices);
lastIndex = max(indices);
lowBinPer = (frequencies(firstIndex)-alphaLow_freq)/(frequencies(firstIndex)-frequencies(firstIndex-1));
lowCorPower = lowBinPer*(frequencies(firstIndex)-frequencies(firstIndex-1))*(.5*abs(magnitude(firstIndex)+magnitude(firstIndex-1)));
highBinPer = (alphaHigh_freq-frequencies(lastIndex))/(frequencies(lastIndex+1)-frequencies(lastIndex));
highCorPower = highBinPer*(frequencies(lastIndex)-frequencies(lastIndex-1))*(.5*abs(magnitude(lastIndex)+magnitude(lastIndex+1)));

%Power Calculation
[~, alphaPower] = msvF(fft_filtered_alpha, T, dt);
alphaPower = alphaPower + highCorPower +lowCorPower;

% Beta band filtered data

% Define the bandpass filter in the frequency domain
betaLow_freq = 12;           % Low cutoff frequency (Hz)
betaHigh_freq = 30;          % High cutoff frequency (Hz)

% Create the frequency-domain filter mask
betaFilter_mask = (frequencies >= betaLow_freq) & (frequencies <= betaHigh_freq);

% Apply the filter to the FFT data
fft_filtered_beta = magnitude(1:end) .* betaFilter_mask;

%Correction for bins not being exactly on cutoffs
indices = find(betaFilter_mask == 1);
firstIndex = min(indices);
lastIndex = max(indices);
lowBinPer = (frequencies(firstIndex)-betaLow_freq)/(frequencies(firstIndex)-frequencies(firstIndex-1));
lowCorPower = lowBinPer*(frequencies(firstIndex)-frequencies(firstIndex-1))*(.5*abs(magnitude(firstIndex)+magnitude(firstIndex-1)));
highBinPer = (betaHigh_freq-frequencies(lastIndex))/(frequencies(lastIndex+1)-frequencies(lastIndex));
highCorPower = highBinPer*(frequencies(lastIndex)-frequencies(lastIndex-1))*(.5*abs(magnitude(lastIndex)+magnitude(lastIndex+1)));

[~, betaPower] = msvF(fft_filtered_beta, T, dt);
betaPower = betaPower + highCorPower +lowCorPower;

% Low Gamma band filtered data

% Define the bandpass filter in the frequency domain
gammaLLow_freq = 30;           % Low cutoff frequency (Hz)
if sampRate/2 >60
    gammaLHigh_freq = 60;      % High cutoff frequency (Hz)
else
    gammaLHigh_freq = sampRate/2-.5;
end

% Create the frequency-domain filter mask
gammaLFilter_mask = (frequencies >= gammaLLow_freq) & (frequencies <= gammaLHigh_freq);

% Apply the filter to the FFT data
fft_filtered_gammaL = magnitude(1:end) .* gammaLFilter_mask;

%Correction for bins not being exactly on cutoffs
indices = find(gammaLFilter_mask == 1);
firstIndex = min(indices);
lastIndex = max(indices);
lowBinPer = (frequencies(firstIndex)-gammaLLow_freq)/(frequencies(firstIndex)-frequencies(firstIndex-1));
lowCorPower = lowBinPer*(frequencies(firstIndex)-frequencies(firstIndex-1))*(.5*abs(magnitude(firstIndex)+magnitude(firstIndex-1)));
highBinPer = (gammaLHigh_freq-frequencies(lastIndex))/(frequencies(lastIndex+1)-frequencies(lastIndex));
highCorPower = highBinPer*(frequencies(lastIndex)-frequencies(lastIndex-1))*(.5*abs(magnitude(lastIndex)+magnitude(lastIndex+1)));

[~, gammaLPower] = msvF(fft_filtered_gammaL, T, dt);
gammaLPower = gammaLPower + highCorPower +lowCorPower;

%60Hz cutout 
% Apply the filter to the FFT data
if sampRate/2 > 60
    fft_filtered_gammaL = magnitude_filtered(1:end) .* gammaLFilter_mask;
    
    % High Gamma band filtered data
    
    % Define the bandpass filter in the frequency domain
    gammaHLow_freq = 60;           % Low cutoff frequency (Hz)
    gammaHHigh_freq = 200;          % High cutoff frequency (Hz)
    
    % Create the frequency-domain filter mask
    gammaHFilter_mask = (frequencies >= gammaHLow_freq) & (frequencies <= gammaHHigh_freq);
    
    % Apply the filter to the FFT data
    fft_filtered_gammaH = magnitude(1:end) .* gammaHFilter_mask;
    
    %Correction for bins not being exactly on cutoffs
    indices = find(gammaHFilter_mask == 1);
    firstIndex = min(indices);
    lastIndex = max(indices);
    lowBinPer = (frequencies(firstIndex)-gammaHLow_freq)/(frequencies(firstIndex)-frequencies(firstIndex-1));
    lowCorPower = lowBinPer*(frequencies(firstIndex)-frequencies(firstIndex-1))*(.5*abs(magnitude(firstIndex)+magnitude(firstIndex-1)));
    highBinPer = (gammaHHigh_freq-frequencies(lastIndex))/(frequencies(lastIndex+1)-frequencies(lastIndex));
    highCorPower = highBinPer*(frequencies(lastIndex)-frequencies(lastIndex-1))*(.5*abs(magnitude(lastIndex)+magnitude(lastIndex+1)));
    
    [~, gammaHPower] = msvF(fft_filtered_gammaH, T, dt);
    gammaHPower = gammaHPower + highCorPower +lowCorPower;
    
    
    %60Hz cutout 
    % Apply the filter to the FFT data
    fft_filtered_gammaH = magnitude_filtered(1:end) .* gammaHFilter_mask;
    
    % High Frequency Oscillations (200+)
    hf_low = 200;
    hf_high = 500; % do we want to go up to the nyquist? That's like 10,000
    
    % Create the frequency-domain filter mask
    hfOscFilter_mask = (frequencies >= hf_low) & (frequencies <= hf_high);
    
    % Apply filter to fft data
    fft_filtered_hfOsc = magnitude(1:end) .* hfOscFilter_mask;
    
    %Correction for bins not being exactly on cutoffs
    indices = find(hfOscFilter_mask == 1);
    firstIndex = min(indices);
    lastIndex = max(indices);
    lowBinPer = (frequencies(firstIndex)-hf_low)/(frequencies(firstIndex)-frequencies(firstIndex-1));
    lowCorPower = lowBinPer*(frequencies(firstIndex)-frequencies(firstIndex-1))*(.5*abs(magnitude(firstIndex)+magnitude(firstIndex-1)));
    highBinPer = (hf_high-frequencies(lastIndex))/(frequencies(lastIndex+1)-frequencies(lastIndex));
    highCorPower = highBinPer*(frequencies(lastIndex)-frequencies(lastIndex-1))*(.5*abs(magnitude(lastIndex)+magnitude(lastIndex+1)));
    
    [~, hfOscPower] = msvF(fft_filtered_hfOsc, T, dt);
    hfOscPower = hfOscPower + highCorPower + lowCorPower;
    
    % Total band filtered data
    
    % Define the bandpass filter in the frequency domain
    totLow_freq = 1;           % Low cutoff frequency (Hz)
    totHigh_freq = 500;          % High cutoff frequency (Hz)
    
    % Create the frequency-domain filter mask
    totFilter_mask = (frequencies >= totLow_freq) & (frequencies <= totHigh_freq);
    
    % Apply the filter to the FFT data
    fft_filtered_tot = magnitude(1:end) .* totFilter_mask;
    
    %Correction for bins not being exactly on cutoffs
    indices = find(totFilter_mask == 1);
    firstIndex = min(indices);
    lastIndex = max(indices);
    lowBinPer = (frequencies(firstIndex)-totLow_freq)/(frequencies(firstIndex)-frequencies(firstIndex-1));
    lowCorPower = lowBinPer*(frequencies(firstIndex)-frequencies(firstIndex-1))*(.5*abs(magnitude(firstIndex)+magnitude(firstIndex-1)));
    highBinPer = (totHigh_freq-frequencies(lastIndex))/(frequencies(lastIndex+1)-frequencies(lastIndex));
    highCorPower = highBinPer*(frequencies(lastIndex)-frequencies(lastIndex-1))*(.5*abs(magnitude(lastIndex)+magnitude(lastIndex+1)));
    
    [~, totbPower] = msvF(fft_filtered_tot, T, dt);
    totbPower = totbPower + highCorPower +lowCorPower;
    
    %60HzCut
    fft_filtered_tot = magnitude_filtered(1:end) .* totFilter_mask;
else
    gammaHPower = 0;
    totbPower = 0;
    hfOscPower = 0;
    fft_filtered_tot = 0;
    fft_filtered_hfOsc = 0;
    fft_filtered_gammaH = 0;
end
% Plot the original and filtered signals in the frequency domain


% figure;
% if sampRate/2>60
%     plot(frequencies, log(abs(fft_filtered_tot)), 'k'); hold on;
%     plot(frequencies, log(abs(fft_filtered_gammaH)), 'b'); hold on;
%     plot(frequencies, log(abs(fft_filtered_hfOsc)), 'Color', '#A2142F'); hold on;
% end
% plot(frequencies, log(abs(fft_filtered_delta)), 'r'); hold on;
% plot(frequencies, log(abs(fft_filtered_theta)), 'm'); hold on;
% plot(frequencies, log(abs(fft_filtered_alpha)), 'y'); hold on;
% plot(frequencies, log(abs(fft_filtered_beta)), 'c'); hold on;
% plot(frequencies, log(abs(fft_filtered_gammaL)), 'g'); hold on;
% 
% titleSpec = ['Total Band Data (Hz)   ' Desc];
% title(titleSpec) 
% xlabel('Frequency (Hz)');
% ylabel('log(Power)');
% if sampRate/2>60
%     xlim([(totLow_freq-1),(totHigh_freq +1)]);
% else
%     xlim([1,sampRate/2])
% end
% grid on;
% hold off;
% 
% if sampRate/2>60
%     Spec_Data = [fft_filtered_tot,fft_filtered_delta,fft_filtered_theta,fft_filtered_alpha,fft_filtered_beta,fft_filtered_gammaL,fft_filtered_gammaH,fft_filtered_hfOsc, frequencies];
% else
%     Spec_Data = [fft_filtered_delta,fft_filtered_theta,fft_filtered_alpha,fft_filtered_beta,fft_filtered_gammaL, frequencies];
% end



%%Bar graph for comparing powers:
%Power outside bands:
%remainderPower = totbPower - (deltaPower+ thetaPower+ alphaPower+ betaPower+ gammaLPower+ gammaHPower + hfOscPower);


% Create sample data
categories = {'Total Band Power','Delta', 'Theta', 'Alpha', 'Beta', 'Low Gamma' 'High Gamma', 'High Frequency Oscillations'};%, 'Outside of Bands'};
values = [totbPower,deltaPower, thetaPower, alphaPower, betaPower, gammaLPower, gammaHPower, hfOscPower];%, remainderPower];

%Normalization
intTime = start-finish;
ambTime = amb_finish-amb_start;

normIntPower = values/intTime;
normAmbPower = ambPowers/ambTime;

normEventPower = zeros(1,length(values));
for k = 1:length(values)
    normEventPower(k) = normIntPower(k)/normAmbPower(k);
end

% Create a bar graph
figure;
bar_handle = bar(values);
% Customize the color of each individual bar
set(gca,'xticklabel',categories);
xlabel('Frequency Bands');
ylabel('Normalized Power');
title(['Power  of Frequency Bands    ']);
grid on;

%%Summary table
powerTable = (transpose([categories; num2cell(values)]));
powerTableCell = cellfun(@num2str, powerTable, 'UniformOutput', false);

%Spectrogram
T = dt * length(interesting_signal(:,2));
nsc = floor(length(interesting_signal(:,2)) / T); % sets the size of each frame in the spectrogram; (1 sec)
nov = floor(nsc / 2); % sets the overlap (divide by 2 so 50% overlap)
nff = max(256, 2^nextpow2(nsc));
[S, F, T] = spectrogram(interesting_signal(:,2), hamming(nsc), nov, nff, sampRate);
imagesc(T, F, 10 * log10(abs(S)));
axis xy;
xlabel('Time (s)');
ylabel('Frequency (Hz)');
title('Spectrogram');
colorbar;

end