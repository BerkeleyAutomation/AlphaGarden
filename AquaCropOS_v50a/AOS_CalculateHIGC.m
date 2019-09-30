function [HIGC,tHI] = AOS_CalculateHIGC(Crop)
% Function to calculate harvest index growth coefficient 

%% Determine HIGC %%
% Total  yield formation days
tHI = Crop.YldFormCD;
% Iteratively estimate HIGC
HIGC = 0.001;
HIest = 0;
while HIest <= (0.98*Crop.HI0)
    HIGC = HIGC+0.001;
    HIest = (Crop.HIini*Crop.HI0)/(Crop.HIini+(Crop.HI0-Crop.HIini)*...
        exp(-HIGC*tHI));
end
if HIest >= Crop.HI0
    HIGC = HIGC-0.001;
end
    
end

