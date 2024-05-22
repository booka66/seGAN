function [intersting_signal, msv] = msvF(signal, T, dt)
%MSV return Mean Squared Value and compute single sided power spectrum
%density
%   Input is the signal in the time domain with two columns (time and voltage)
    df = 1 / T;
    intersting_signal = signal * dt;

    INTERESTING_SIGNAL_psd = abs(intersting_signal).^2 * (1 / T);

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