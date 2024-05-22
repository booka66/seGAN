function [INTERESTING_SIGNAL, msv] = MSV(signal, fs)
%MSV return Mean Squared Value and compute single sided power spectrum
%density
%   Input is the signal in the time domain with two columns (time and voltage)
    dt = 1 / fs;
    T = dt * length(signal);
    df = 1 / T;
    INTERESTING_SIGNAL = fft(signal) * dt;

    INTERESTING_SIGNAL_psd = abs(INTERESTING_SIGNAL).^2 * (1 / T);

    % Get the real half of the signal; in both cases, we don't want the 0
    % frequency, but is it only one case where we want to cut out the
    % Nyquist? or is that both cases?
    
    if mod(length(INTERESTING_SIGNAL_psd), 2) == 1 
        INTERESTING_SIGNAL_psd = INTERESTING_SIGNAL_psd(1:ceil(length(INTERESTING_SIGNAL_psd)/2));
        INTERESTING_SIGNAL_psd(2:end) = 2 * INTERESTING_SIGNAL_psd(2: end);
    else
        INTERESTING_SIGNAL_psd = INTERESTING_SIGNAL_psd(1:length(INTERESTING_SIGNAL_psd) / 2);
        INTERESTING_SIGNAL_psd(2:end-1) = 2 * INTERESTING_SIGNAL_psd(2: end-1);
    end
    % sum(gxxi * deltaf)
    msv = sum(INTERESTING_SIGNAL_psd) * (df);
end