function [thdry,thwp,thfc,ths,ksat] = AOS_SoilHydraulicProperties(Soil)
% Function to calculate soil hydraulic properties, given textural inputs.
% Calculations use pedotransfer function equations described in Saxton and
% Rawls (2006)

%% Calculate soil hydraulic properties %%
% Initialise values
[thdry,thwp,thfc,ths,ksat] = deal(zeros(1,Soil.nLayer));
% Do calculations for each soil layer
for ii = 1:Soil.nLayer
    % Water content at permanent wilting point
    Pred_thWP = -(0.024*Soil.Layer.Sand(ii))+(0.487*Soil.Layer.Clay(ii))+...
        (0.006*Soil.Layer.OrgMat(ii))+(0.005*(Soil.Layer.Sand(ii)*...
        Soil.Layer.OrgMat(ii)))-(0.013*(Soil.Layer.Clay(ii)*Soil.Layer.OrgMat(ii)))+...
        (0.068*(Soil.Layer.Sand(ii)*Soil.Layer.Clay(ii)))+0.031;
    
    thwp(ii) = Pred_thWP+(0.14*Pred_thWP)-0.02;
    
    % Water content at field capacity and saturation
    Pred_thFC = (-0.251*Soil.Layer.Sand(ii))+(0.195*Soil.Layer.Clay(ii))+...
        (0.011*Soil.Layer.OrgMat(ii))+(0.006*(Soil.Layer.Sand(ii)*...
        Soil.Layer.OrgMat(ii)))-(0.027*(Soil.Layer.Clay(ii)*Soil.Layer.OrgMat(ii)))+...
        (0.452*(Soil.Layer.Sand(ii)*Soil.Layer.Clay(ii)))+0.299;
    
    PredAdj_thFC = Pred_thFC+((1.283*(Pred_thFC^2))-(0.374*Pred_thFC)-0.015);
    
    Pred_thS33 = (0.278*Soil.Layer.Sand(ii))+(0.034*Soil.Layer.Clay(ii))+...
        (0.022*Soil.Layer.OrgMat(ii))-(0.018*(Soil.Layer.Sand(ii)*...
        Soil.Layer.OrgMat(ii)))-(0.027*(Soil.Layer.Clay(ii)*Soil.Layer.OrgMat(ii)))-...
        (0.584*(Soil.Layer.Sand(ii)*Soil.Layer.Clay(ii)))+0.078;
    
    PredAdj_thS33 = Pred_thS33+((0.636*Pred_thS33)-0.107);
    
    Pred_thS = (PredAdj_thFC+PredAdj_thS33)+((-0.097*Soil.Layer.Sand(ii))+0.043);
    
    pN = (1-Pred_thS)*2.65;
    pDF = pN*Soil.Layer.DF(ii);
    PorosComp = (1-(pDF/2.65))-(1-(pN/2.65));
    PorosCompOM = 1-(pDF/2.65);
    
    DensAdj_thFC = PredAdj_thFC+(0.2*PorosComp);
    DensAdj_thS = PorosCompOM;
    
    thfc(ii) = DensAdj_thFC;
    ths(ii) = DensAdj_thS;
    
    % Saturated hydraulic conductivity (mm/day)
    lambda = 1/((log(1500)-log(33))/(log(thfc(ii))-log(thwp(ii))));
    ksat(ii) = (1930*(ths(ii)-thfc(ii))^(3-lambda))*24;
    
    % Water content at air dry
    thdry(ii) = thwp(ii)/2;
    
    % Round values
    thdry(ii) = round((10000*thdry(ii)))/10000;
    thwp(ii) = round((1000*thwp(ii)))/1000;
    thfc(ii) = round((1000*thfc(ii)))/1000;
    ths(ii) = round((1000*ths(ii)))/1000;
    ksat(ii) = round((10*ksat(ii)))/10;
end

end

