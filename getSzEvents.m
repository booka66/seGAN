function segments = getSzEvents(logical_list)
    segments = [];
    start = [];
    
    for i = 1:length(logical_list)
        if logical_list(i) && isempty(start)
            start = i;
        elseif ~logical_list(i) && ~isempty(start)
            segments = [segments; [start, i - 1]];
            start = [];
        end
    end
    
    % Check if the last segment extends to the end of the list
    if ~isempty(start)
        segments = [segments; [start, length(logical_list)]];
    end
end