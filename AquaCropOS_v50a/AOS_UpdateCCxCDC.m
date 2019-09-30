function [CCXadj,CDCadj] = AOS_UpdateCCxCDC(CCprev,CDC,CCx,dt)
% Function to update CCx and CDC parameter valyes for rewatering in late
% season of an early declining canopy

%% Get adjusted CCx %%
CCXadj = CCprev/(1-0.05*(exp(dt*(CDC/CCx))-1));

%% Get adjusted CDC %%
CDCadj = CDC*(CCXadj/CCx);

end

