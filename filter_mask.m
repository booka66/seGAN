function [NaN_ind] = filter_mask(frequencies, center_frequency)
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here
bw = frequencies(2) - frequencies(1);
low_bound = center_frequency - 10*bw;
upper_bound = center_frequency + 10*bw;
mask = (frequencies < low_bound) | (upper_bound <= frequencies);
NaN_ind = mask == 0;
end