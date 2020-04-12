function [AOS_ClockStruct,AOS_InitialiseStruct] = AOS_CheckModelTermination(AOS_ClockStruct,...
    AOS_InitialiseStruct)
% Function to check and declare model termination

%% Check if current time-step is the last
CurrentTime = AOS_ClockStruct.StepEndTime;
if CurrentTime < AOS_ClockStruct.SimulationEndDate
    AOS_ClockStruct.ModelTermination = false;
elseif CurrentTime >= AOS_ClockStruct.SimulationEndDate
    AOS_ClockStruct.ModelTermination = true;
end

%% Check if at the end of last growing season %%
% Allow model to exit early if crop has reached maturity or died, and in
% the last simulated growing season
if (AOS_InitialiseStruct.InitialCondition.HarvestFlag == true) &&...
            (AOS_ClockStruct.SeasonCounter == AOS_ClockStruct.nSeasons)
        AOS_ClockStruct.ModelTermination = true;
end

end

