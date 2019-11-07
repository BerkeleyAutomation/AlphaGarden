function [tReq] = AOS_CCRequiredTime(CCprev,CCo,CCx,CGC,CDC,dt,tSum,Mode)
% Function to find time required to reach CC at end of previous day, given
% current CGC or CDC

%% Get CGC and/or time (GDD or CD) required to reach CC on previous day %%
if strcmp(Mode,'CGC')
    if CCprev <= (CCx/2)
        CGCx = (log(CCprev/CCo))/(tSum-dt);
    else
        CGCx = (log((0.25*CCx*CCx/CCo)/(CCx-CCprev)))/(tSum-dt);
    end
    tReq = (tSum-dt)*(CGCx/CGC);
elseif strcmp(Mode,'CDC');
    tReq = (log(1+(1-CCprev/CCx)/0.05))/(CDC/CCx);
end
end

