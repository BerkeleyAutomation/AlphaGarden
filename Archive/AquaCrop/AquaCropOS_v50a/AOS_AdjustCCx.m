function [CCxAdj] = AOS_AdjustCCx(CCprev,CCo,CCx,CGC,CDC,dt,tSum,Crop)
% Function to adjust CCx value for changes in CGC due to water stress
% during the growing season

%% Get time required to reach CC on previous day %%
tCCtmp = AOS_CCRequiredTime(CCprev,CCo,CCx,CGC,CDC,dt,tSum,'CGC');

%% Determine CCx adjusted %%
if tCCtmp > 0
    tCCtmp = tCCtmp+(Crop.CanopyDevEnd-tSum)+dt;
    CCxAdj = AOS_CCDevelopment(CCo,CCx,CGC,CDC,tCCtmp,'Growth');
else
    CCxAdj = 0;
end

end

