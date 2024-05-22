function [SzEventsTimes, SE_list, Sz_power_list] = getSzEnvelop_wSE(Vdata,sampRate,t)
%Seizure Detection
vartol = 12; %tolerance for variance
freqtol = 8; %tolerance for frequency
varwindowSize = .25; %initial variance windowing
Ref_window_time = 120; %ref window time
coherencewindow = 2; %window in which Var and Freq have to coincide in sec
threshold_seconds = 7; %same event limit
VlimPer = 2; %percent of baseline SD that sets outlier lims
minVoutReq = round(sampRate); %1 second
freqLowLim = 5; %high pass lower limit (Hz)

SzEventsTimes = [];
traceVariance = transpose(rolling_variance(Vdata, sampRate));

if sampRate/2 > 150
    frequency_range = [freqLowLim,150]; % Frequency range of interest in Hz
else
    frequency_range = [freqLowLim,round(sampRate/2)]; % Frequency range of interest in Hz
end
window_size = round(sampRate*varwindowSize); % Specify the rolling window size
rolling_power = calculatePower(Vdata, sampRate, frequency_range, window_size, 0);% Calculate rolling power for the specified frequency range
t2 = transpose(linspace(0,length(Vdata)/sampRate, length(rolling_power)));

%REFERENCE:
%-Choose reference section (1 min) (in first 10min)  with:
%	-no LFP activity
%	-no seizure-like activity
%	-no electrical noise spikes
%-Set the reference section as spectral activity baseline and voltage noise floor

% Set the window size to be for one minute (sampling rate*60sec)
window_size1 = round(sampRate*Ref_window_time);
Vfirst10 = round(sampRate*60*10);

% Specify the step size for the window
step_size = round(sampRate/2);

% Initialize
range = 1:step_size:(Vfirst10 - window_size1 + 1);
min_dev_values = zeros(1, numel(range));

% Calculate deviation in each window
for i = 1:numel(range)
    end_index = range(i) + window_size1 - 1;

    % Check if the end index exceeds the size of V
    if end_index > numel(Vdata)
        break;  % Exit the loop if the index is out of bounds
    end

    window_data = Vdata(range(i):end_index);
    deviation = abs(window_data - mean(window_data));
    min_dev_values(i) = sum(deviation);
end

% Find the minimum value and its index
[~, min_index] = min(min_dev_values);

% Use median group
middle_idx = ceil(length(min_dev_values)/2);
min_index = middle_idx;

% Store the reference section location
referenceIdxs = [range(min_index), (range(min_index) + window_size1 - 1)];

%Store the reference section as a unique list
VRef = Vdata(referenceIdxs(1):referenceIdxs(2));
%tRef = t(referenceIdxs(1):referenceIdxs(2));
power_ref = calculatePower(VRef, sampRate, frequency_range, window_size, 0);
varianceRef = transpose(rolling_variance(VRef, sampRate));

VlimUpper = mean(VRef)+ VlimPer*std(VRef);
VlimLower = mean(VRef)- VlimPer*std(VRef);

%combine Variance and Power coherence

fitVar = fitListSize(traceVariance, rolling_power);

variThresh = mean(varianceRef)+vartol*std(varianceRef);
powThresh = mean(power_ref)+freqtol*std(power_ref);
oneSec = find(t2>1,1);
Sz_window_size = round(oneSec*coherencewindow);

szList = rollingWindowThreshold(fitVar, rolling_power, Sz_window_size, variThresh, powThresh);

%Cut out any seizures less than 10sec
szTimeMin = 10; %in sec
szListClean = szLenLim(szList,oneSec*szTimeMin);

%extract the event details for the seizure list
SzEventsIdxs = getSzEvents(szListClean);
SzEventTimes = t2(SzEventsIdxs);

% Initialize the merged events matrix
merged_events = [];
Sz_power_list = [];


%Check if any Szs detected
if isempty(SzEventTimes)
    SzEventsTimes = 0;
    SE_list = 0;
else
    if size(SzEventTimes,2) == 1
        SzEventsTimes = SzEventTimes;
        %%remove low V amplitude events
        qualified_time_list = []; % Initialize an empty list to store qualified intervals
 
        start_time = SzEventTimes(1); % Get the start time for this interval
        end_time = SzEventTimes(2); % Get the end time for this interval
        start_idx = find(t); % Get the start time for this interval
        end_idx = SzEventsIdxs(2)+ round(20*sampRate); % Get the end time for this interval
        
        % Create a subgroup of the voltage data for the current interval
        voltage_subgroup = Vdata(round(start_idx):round(end_idx));
        
        % Check if there are at least 2 values greater than Vfloor
        num_greater_than_Vfloor = sum(voltage_subgroup > VlimUpper) >= minVoutReq;
        
        % Check if there are at least 2 values less than Vceil
        num_less_than_Vceil = sum(voltage_subgroup < VlimLower) >= minVoutReq;
        
        % If both conditions are met, add the interval to the qualified list
        if num_greater_than_Vfloor || num_less_than_Vceil
            qualified_time_list = [qualified_time_list; start_time, end_time];
        end
        
        SzEventTimes = qualified_time_list;
    end

    if size(SzEventTimes,2) > 1
        %Exception for a size of 1 event
        %%remove low V amplitude events
        qualified_time_list = []; % Initialize an empty list to store qualified intervals
        
        for i = 1:size(SzEventTimes, 1) % Loop through each row of the time_list
            start_time = SzEventTimes(i, 1); % Get the start time for this interval
            end_time = SzEventTimes(i, 2); % Get the end time for this interval
            start_idx = find(t>start_time,1); % Get the start time for this interval
            end_idx = find(t>end_time,1); % Get the end time for this interval
            
            % Create a subgroup of the voltage data for the current interval
            voltage_subgroup = Vdata(round(start_idx):round(end_idx));
           
            % Check if there are at least 2 values greater than Vfloor
            num_greater_than_Vfloor = sum(voltage_subgroup > VlimUpper);

            % Check if there are at least 2 values less than Vceil
            num_less_than_Vceil = sum(voltage_subgroup < VlimLower);

            % If both conditions are met, add the interval to the qualified list
            if (num_greater_than_Vfloor + num_less_than_Vceil)>= minVoutReq
                qualified_time_list = [qualified_time_list; start_time, end_time];
            end
        end
        
        SzEventTimes = qualified_time_list;
        % Iterate over rows of the events matrix
        if length(SzEventTimes)>0

            current_event = SzEventTimes(1, :);
            for i = 2:size(SzEventTimes, 1)
                % Check if the end time of the previous event is within the threshold
                if SzEventTimes(i, 1) - current_event(2) <= threshold_seconds
                    % Merge the current event with the previous one
                    current_event(2) = SzEventTimes(i, 2);
                else
                    % Add the merged event to the merged events matrix
                    merged_events = [merged_events; current_event];
                    % Set the current event to the next event
                    current_event = SzEventTimes(i, :);
                end
            end
            SzEventsTimes = [merged_events; current_event];
        end
    end
    
    
        % Initialize an empty list to store the filtered rows
        SE_list = [];

        % Iterate over each row in the time_list
        if size(SzEventTimes, 2) < 2
            if isempty(SzEventTimes)
                placeholder = 1;
            elseif (SzEventTimes(2)- SzEventTimes(1)) > 5*60
                SE_list = SzEventTimes;
            end
        else
            for i = 1:size(SzEventsTimes, 1)
                start_time = SzEventsTimes(i, 1);
                end_time = SzEventsTimes(i, 2);
                stIdx = find(t > start_time+60, 1);
                fnIdx = find(t > end_time+60, 1);
                if isempty(fnIdx)
                    fnIdx = length(t);
                end
                voltage_subgroup = Vdata(stIdx:fnIdx);
                rang = max(voltage_subgroup)-min(voltage_subgroup);

                Sz_pwr = freqPowerPercentage(voltage_subgroup, VRef, frequency_range, sampRate);%TODO: make this 5-150Hz power
                Sz_power_list = [Sz_power_list Sz_pwr];


                % Check if the time interval is greater than the threshold
                if (end_time-start_time) > 5*60
                    % If the time interval is greater than 5 minutes, add the row to the filtered_list
                    SE_list = [SE_list; SzEventsTimes(i, :)];
                end
            end
        end
        
        %Within SE, combine discharges within 6 min
        if size(SzEventTimes,2)>1
            for i = 2:size(SzEventsTimes, 1)
                for j = 1:size(SE_list,1)
                    if SzEventsTimes(i, 1) > SE_list(j, 2)
                        if (SzEventsTimes(i, 1) - SE_list(j,2)) < 6*60
                            SE_list(j,2) = SzEventsTimes(i,2);
                        end
                    end
                end
            end
            
            if size(SE_list, 1) >1
                for i = 1:size(SE_list, 1)-1
                    if SE_list(i,2) >= SE_list(i,2)
                        SE_list(i+1,:) = SE_list(i,:);
                    end
                end
            end
        
            %Remove Duplicates
            [~, unique_indices, ~] = unique(SE_list, 'rows');
            SE_list = SE_list(unique_indices, :);
        end
end
   
end
