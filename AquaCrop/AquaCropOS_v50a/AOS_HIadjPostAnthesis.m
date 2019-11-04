function [NewCond] = AOS_HIadjPostAnthesis(InitCond,Crop,Ksw)
% Function to calculate adjustment to harvest index for post-anthesis water
% stress

%% Store initial conditions in a structure for updating %%
NewCond = InitCond;

%% Calculate harvest index adjustment %%
% 1. Adjustment for leaf expansion
tmax1 = Crop.CanopyDevEndCD-Crop.HIstartCD;
DAP = NewCond.DAP-InitCond.DelayedCDs;
if (DAP <= (Crop.CanopyDevEndCD+1)) && (tmax1 > 0) &&...
        (NewCond.Fpre > 0.99) && (NewCond.CC > 0.001) &&...
        (Crop.a_HI > 0)
    dCor = (1+(1-Ksw.Exp)/Crop.a_HI);
    NewCond.sCor1 = InitCond.sCor1+(dCor/tmax1);
    DayCor = DAP-1-Crop.HIstartCD;
    NewCond.fpost_upp = (tmax1/DayCor)*NewCond.sCor1;
end

% 2. Adjustment for stomatal closure
tmax2 = Crop.YldFormCD;
DAP = NewCond.DAP-InitCond.DelayedCDs;
if (DAP <= (Crop.HIendCD+1)) && (tmax2 > 0) &&...
        (NewCond.Fpre > 0.99) && (NewCond.CC > 0.001) &&...
        (Crop.b_HI > 0)
    dCor = (exp(0.1*log(Ksw.Sto)))*(1-(1-Ksw.Sto)/Crop.b_HI);
    NewCond.sCor2 = InitCond.sCor2+(dCor/tmax2);
    DayCor = DAP-1-Crop.HIstartCD;
    NewCond.fpost_dwn = (tmax2/DayCor)*NewCond.sCor2;
end

% Determine total multiplier
if (tmax1 == 0) && (tmax2 == 0)
    NewCond.Fpost = 1;
else
    if tmax2 == 0
        NewCond.Fpost = NewCond.fpost_upp;
    else
        if tmax1 == 0
            NewCond.Fpost = NewCond.fpost_dwn;
        elseif tmax1 <= tmax2
            NewCond.Fpost = NewCond.fpost_dwn*(((tmax1*NewCond.fpost_upp)+...
                (tmax2-tmax1))/tmax2);
        else
            NewCond.Fpost = NewCond.fpost_upp*(((tmax2*NewCond.fpost_dwn)+...
                (tmax1-tmax2))/tmax1);
        end
    end
end
    
end

