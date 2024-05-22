function powerPercentage = freqPowerPercentage(voltage_subgroup, VRef, frequency_range, sampRate)
    if isempty(voltage_subgroup)
        powerPercentage = 0;
    else
        % Calculate the FFT of the voltage subgroup
        Nsub = length(voltage_subgroup);
        Xsub = fft(voltage_subgroup, Nsub);
        Xsub = Xsub(1:round(Nsub/2+1));
        power_subgroup = abs(Xsub).^2;
        t_subgroup = length(voltage_subgroup)*(1/sampRate);
        
        % Calculate the FFT of the reference signal
        Nref = length(VRef);
        Xref = fft(VRef, Nref);
        Xref = Xref(1:round(Nref/2+1));
        power_ref = abs(Xref).^2;
        t_ref = length(VRef)*(1/sampRate);
    
        % Determine the frequency indices for the given range for the subgroup
        freqSub = sampRate * (0:Nsub/2)/Nsub;
        lower_idxSub = find(freqSub >= frequency_range(1), 1, 'first');
        upper_idxSub = find(freqSub <= frequency_range(2), 1, 'last');
        
        % Determine the frequency indices for the given range for the reference
        freqRef = sampRate * (0:Nref/2)/Nref;
        lower_idxRef = find(freqRef >= frequency_range(1), 1, 'first');
        upper_idxRef = find(freqRef <= frequency_range(2), 1, 'last');
    
        % Calculate the sum of frequency powers in the given range
        subgroup_power_range = sum(power_subgroup(lower_idxSub:upper_idxSub));
        ref_power_range = sum(power_ref(lower_idxRef:upper_idxRef));
        
        % Calculate the percentage of power in the given range
        powerPercentage = ((subgroup_power_range/t_subgroup) / (ref_power_range/t_ref)) * 100;
    end
end
