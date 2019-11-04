function [Ksw] = AOS_OutputWaterStress(Crop,Soil,InitCond,Et0)
% Function to output water stress levels

% Calculate root zone water content
[~,Dr,TAW,~] = AOS_RootZoneWater(Soil,Crop,InitCond);

% Determine if water stress is occurring
beta = true;
Ksw = AOS_WaterStress(Crop,InitCond,Dr,TAW,Et0,beta);

end