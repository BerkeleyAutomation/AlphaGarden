function [] = AOS_UpdateTime()
% Function to update current time in model

%% Define global variables %%
global AOS_ClockStruct
global AOS_InitialiseStruct

%% Update time %%
if AOS_ClockStruct.ModelTermination == false
    if (AOS_InitialiseStruct.InitialCondition.HarvestFlag == true) &&...
            (strcmp(AOS_ClockStruct.OffSeason,'N'))
        % End of growing season has been reached and not simulating
        % off-season soil water balance. Advance time to the start of the
        % next growing season.
        % Check if in last growing season 
        if AOS_ClockStruct.SeasonCounter < AOS_ClockStruct.nSeasons
            % Update growing season counter
            AOS_ClockStruct.SeasonCounter = AOS_ClockStruct.SeasonCounter+1;
            % Update time-step counter
            AOS_ClockStruct.TimeStepCounter = find(AOS_ClockStruct.TimeSpan==...
                AOS_ClockStruct.PlantingDate(AOS_ClockStruct.SeasonCounter));
            % Update start time of time-step
            AOS_ClockStruct.StepStartTime = ...
                AOS_ClockStruct.TimeSpan(AOS_ClockStruct.TimeStepCounter);
            % Update end time of time-step
            AOS_ClockStruct.StepEndTime = ...
                AOS_ClockStruct.TimeSpan(AOS_ClockStruct.TimeStepCounter+1);
            % Reset initial conditions for start of growing season
            AOS_ResetInitialConditions();
        end
    else
        % Simulation considers off-season, so progress by one time-step
        % (one day)
        % Time-step counter
        AOS_ClockStruct.TimeStepCounter = AOS_ClockStruct.TimeStepCounter+1;
        % Start of time step (beginning of current day)
        AOS_ClockStruct.StepStartTime = ...
            AOS_ClockStruct.TimeSpan(AOS_ClockStruct.TimeStepCounter);
        % End of time step (beginning of next day)
        AOS_ClockStruct.StepEndTime = ...
            AOS_ClockStruct.TimeSpan(AOS_ClockStruct.TimeStepCounter+1);
        % Check if in last growing season
        if AOS_ClockStruct.SeasonCounter < AOS_ClockStruct.nSeasons
            % Check if upcoming day is the start of a new growing season
            if AOS_ClockStruct.StepStartTime == ...
                    AOS_ClockStruct.PlantingDate(AOS_ClockStruct.SeasonCounter+1);
                % Update growing season counter
                AOS_ClockStruct.SeasonCounter = AOS_ClockStruct.SeasonCounter+1;
                % Reset initial conditions for start of growing season
                AOS_ResetInitialConditions();
            end
        end
    end
elseif AOS_ClockStruct.ModelTermination == true
    AOS_ClockStruct.StepStartTime = AOS_ClockStruct.StepEndTime;
    AOS_ClockStruct.StepEndTime = AOS_ClockStruct.StepEndTime+1;
end

end

