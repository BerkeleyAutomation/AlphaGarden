function [NewCond,Irr] = AOS_Irrigation(InitCond,GrowingSeason,Irr)
% Function to update irrigation for current day

%% Store intial conditions for updating %%
NewCond = InitCond;
        
%% Determine irrigation depth (mm/day) to be applied %%
if GrowingSeason == true
    % Update growth stage if it is first day of a growing season
    if NewCond.DAP == 1
        NewCond.GrowthStage = 1;
    end
    % Update cumulative irrigation counter for growing season
    NewCond.IrrCum = NewCond.IrrCum+Irr;
elseif GrowingSeason == false
    % No irrigation outside growing season
    Irr = 0;
    NewCond.IrrCum = 0;
end

end

