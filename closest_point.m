function [value, index] = closest_point(trace,point)
    abs_difference = abs(trace - point);
    index = find(abs_difference == min(abs_difference));
    value = trace(index);
end