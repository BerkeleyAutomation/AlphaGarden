function [NewCond] = AOS_CanopyCover(Crop,Soil,InitCond,GDD,Et0,GrowingSeason)
% Function to simulate canopy growth/decline

%% Store initial conditions in a new structure for updating %%
NewCond = InitCond;
NewCond.CCprev = InitCond.CC;

%% Calculate canopy development (if in growing season) %%
if GrowingSeason == true
    % Calculate root zone water content
    [~,Dr,TAW,~] = AOS_RootZoneWater(Soil,Crop,NewCond);

    % Determine if water stress is occurring
    beta = true;
    Ksw = AOS_WaterStress(Crop,NewCond,Dr,TAW,Et0,beta);

    % Get canopy cover growth time
    if Crop.CalendarType == 1
        tCC = NewCond.DAP;
        dtCC = 1;
        tCCadj = NewCond.DAP-NewCond.DelayedCDs;
    elseif Crop.CalendarType == 2
        tCC = NewCond.GDDcum;
        dtCC = GDD;
        tCCadj = NewCond.GDDcum-NewCond.DelayedGDDs;
    end

    %% Canopy development (potential) %%
    if (tCC < Crop.Emergence) || (round(tCC) > Crop.Maturity)
        % No canopy development before emergence/germination or after
        % maturity
        NewCond.CC_NS = 0;    
    elseif tCC < Crop.CanopyDevEnd
        % Canopy growth can occur
        if (InitCond.CC_NS <= Crop.CC0)
            % Very small initial CC as it is first day or due to senescence.
            % In this case, assume no leaf expansion stress 
            NewCond.CC_NS = Crop.CC0*exp(Crop.CGC*dtCC);
        else
            % Canopy growing
            tmp_tCC = tCC-Crop.Emergence;
            NewCond.CC_NS = AOS_CCDevelopment(Crop.CC0,Crop.CCx,...
                Crop.CGC,Crop.CDC,tmp_tCC,'Growth');
        end
        % Update maximum canopy cover size in growing season
        NewCond.CCxAct_NS = NewCond.CC_NS;   
    elseif tCC > Crop.CanopyDevEnd
        % No more canopy growth is possible or canopy in decline  
        % Set CCx for calculation of withered canopy effects
        NewCond.CCxW_NS = NewCond.CCxAct_NS;
        if tCC < Crop.Senescence
            % Mid-season stage - no canopy growth
            NewCond.CC_NS = InitCond.CC_NS;
            % Update maximum canopy cover size in growing season
            NewCond.CCxAct_NS = NewCond.CC_NS;
        else
            % Late-season stage - canopy decline
            tmp_tCC = tCC-Crop.Senescence;
            NewCond.CC_NS = AOS_CCDevelopment(Crop.CC0,Crop.CCx,...
                Crop.CGC,Crop.CDC,tmp_tCC,'Decline');
        end
    end

    %% Canopy development (actual) %%
    if (tCCadj < Crop.Emergence) || (round(tCCadj) > Crop.Maturity)
        % No canopy development before emergence/germination or after
        % maturity
        NewCond.CC = 0;
    elseif tCCadj < Crop.CanopyDevEnd
        % Canopy growth can occur
        if InitCond.CC <= NewCond.CC0adj
            % Very small initial CC as it is first day or due to senescence. In
            % this case, assume no leaf expansion stress 
            NewCond.CC = NewCond.CC0adj*exp(Crop.CGC*dtCC);
        else
            % Canopy growing
            if InitCond.CC >= (0.9799*Crop.CCx)
                % Canopy apprAOShing maximum size
                tmp_tCC = tCC-Crop.Emergence;
                NewCond.CC = AOS_CCDevelopment(Crop.CC0,Crop.CCx,...
                    Crop.CGC,Crop.CDC,tmp_tCC,'Growth');
                NewCond.CC0adj = Crop.CC0;
            else
                % Adjust canopy growth coefficient for leaf expansion water 
                % stress effects
                CGCadj = Crop.CGC*Ksw.Exp;
                if CGCadj > 0
                    % Adjust CCx for change in CGC
                    CCXadj = AOS_AdjustCCx(InitCond.CC,NewCond.CC0adj,Crop.CCx,...
                        CGCadj,Crop.CDC,dtCC,tCCadj,Crop);
                    if CCXadj > 0
                        if abs(InitCond.CC-Crop.CCx) < 0.00001
                            % ApprAOShing maximum canopy cover size
                            tmp_tCC = tCC-Crop.Emergence;
                            NewCond.CC = AOS_CCDevelopment(Crop.CC0,Crop.CCx,...
                                Crop.CGC,Crop.CDC,tmp_tCC,'Growth');
                        else
                            % Determine time required to reach CC on previous,
                            % day, given CGCAdj value
                            tReq = AOS_CCRequiredTime(InitCond.CC,NewCond.CC0adj,...
                                CCXadj,CGCadj,Crop.CDC,dtCC,tCCadj,'CGC');
                            % Calclate GDD's for canopy growth
                            tmp_tCC = tReq+dtCC;
                            if tmp_tCC > 0
                                % Determine new canopy size
                                NewCond.CC = AOS_CCDevelopment(NewCond.CC0adj,CCXadj,...
                                    CGCadj,Crop.CDC,tmp_tCC,'Growth');
                            else
                                % No canopy growth
                                NewCond.CC = InitCond.CC;
                            end
                        end
                    else
                        % No canopy growth
                        NewCond.CC = InitCond.CC;
                    end
                else
                    % No canopy growth
                    NewCond.CC = InitCond.CC;
                    % Update CC0 if current canopy cover is less than
                    % initial canopy cover size at planting
                    if NewCond.CC < NewCond.CC0adj
                        NewCond.CC0adj = NewCond.CC;
                    end
                end
            end
        end
        if NewCond.CC > InitCond.CCxAct
            % Update actual maximum canopy cover size during growing season
            NewCond.CCxAct = NewCond.CC;
        end
    elseif tCCadj > Crop.CanopyDevEnd
        % No more canopy growth is possible or canopy is in decline
        if tCCadj < Crop.Senescence
            % Mid-season stage - no canopy growth
            NewCond.CC = InitCond.CC;
            if NewCond.CC > InitCond.CCxAct
                % Update actual maximum canopy cover size during growing
                % season
                NewCond.CCxAct = NewCond.CC;
            end
        else
            % Late-season stage - canopy decline
            % Adjust canopy decline coefficient for difference between actual
            % and potential CCx
            CDCadj = Crop.CDC*(NewCond.CCxAct/Crop.CCx);
            % Determine new canopy size
            tmp_tCC = tCCadj-Crop.Senescence;
            NewCond.CC = AOS_CCDevelopment(NewCond.CC0adj,NewCond.CCxAct,...
                Crop.CGC,CDCadj,tmp_tCC,'Decline');
        end
        % Check for crop growth termination
        if (NewCond.CC < 0.001) && (InitCond.CropDead == false)
            % Crop has died
            NewCond.CC = 0;
            NewCond.CropDead = true;
        end
    end

    %% Canopy senescence due to water stress (actual) %%
    if tCCadj >= Crop.Emergence
        if (tCCadj < Crop.Senescence) || (InitCond.tEarlySen > 0)
            % Check for early canopy senescence starting/continuing due to severe
            % water stress
            if Ksw.Sen < 1
                % Early canopy senescence
                NewCond.PrematSenes = true;
                if InitCond.tEarlySen == 0
                    % No prior early senescence
                    NewCond.CCxEarlySen = InitCond.CC;
                end
                % Increment early senescence GDD counter
                NewCond.tEarlySen = InitCond.tEarlySen+dtCC;
                % Adjust canopy decline coefficient for water stress
                beta = false;
                Ksw = AOS_WaterStress(Crop,NewCond,Dr,TAW,Et0,beta);
                if Ksw.Sen > 0.99999
                    CDCadj = 0.0001;
                else
                    CDCadj = (1-(Ksw.Sen^8))*Crop.CDC;
                end
                % Get new canpy cover size after senescence
                if NewCond.CCxEarlySen < 0.001
                    CCsen = 0;
                else
                    % Get time required to reach CC at end of previous day, given
                    % CDCadj
                    tReq = AOS_CCRequiredTime(InitCond.CC,NewCond.CC0adj,...
                        NewCond.CCxEarlySen,Crop.CGC,CDCadj,dtCC,tCCadj,'CDC');
                    % Calculate GDD's for canopy decline
                    tmp_tCC = tReq+dtCC;
                    % Determine new canopy size
                    CCsen = AOS_CCDevelopment(NewCond.CC0adj,NewCond.CCxEarlySen,...
                        Crop.CGC,CDCadj,tmp_tCC,'Decline');
                end

                % Update canopy cover size
                if tCCadj < Crop.Senescence
                    % Limit CC to CCx
                    if CCsen > Crop.CCx
                       CCsen = Crop.CCx;
                    end
                    % CC cannot be greater than value on previous day
                    NewCond.CC = CCsen;
                    if NewCond.CC > InitCond.CC
                        NewCond.CC = InitCond.CC;
                    end
                    % Update maximum canopy cover size during growing
                    % season
                    NewCond.CCxAct = NewCond.CC;
                    % Update CC0 if current CC is less than initial canopy
                    % cover size at planting
                    if NewCond.CC < Crop.CC0
                        NewCond.CC0adj = NewCond.CC;
                    else
                        NewCond.CC0adj = Crop.CC0;
                        
                    end 
                else
                    % Update CC to account for canopy cover senescence due
                    % to water stress
                    if CCsen < NewCond.CC
                        NewCond.CC = CCsen;
                    end
                end
                % Check for crop growth termination
                if (NewCond.CC < 0.001) && (InitCond.CropDead == false)
                    % Crop has died
                    NewCond.CC = 0;
                    NewCond.CropDead = true;
                end     
            else
                % No water stress
                NewCond.PrematSenes = false;
                if (tCCadj > Crop.Senescence) && (InitCond.tEarlySen > 0)
                    % Rewatering of canopy in late season
                    % Get new values for CCx and CDC
                    tmp_tCC = tCCadj-dtCC-Crop.Senescence;
                    [CCXadj,CDCadj] = AOS_UpdateCCxCDC(InitCond.CC,...
                        Crop.CDC,Crop.CCx,tmp_tCC);            
                    % Get new CC value for end of current day
                    tmp_tCC = tCCadj-Crop.Senescence;
                    NewCond.CC = AOS_CCDevelopment(NewCond.CC0adj,CCXadj,...
                        Crop.CGC,CDCadj,tmp_tCC,'Decline');
                    % Check for crop growth termination
                    if (NewCond.CC < 0.001) && (InitCond.CropDead == false)
                        NewCond.CC = 0;
                        NewCond.CropDead = true;
                    end
                end
                % Reset early senescence counter
                NewCond.tEarlySen = 0;
            end
            % Adjust CCx for effects of withered canopy
            if NewCond.CC > InitCond.CCxW
                NewCond.CCxW = NewCond.CC;
            end
        end
    end

    %% Calculate canopy size adjusted for micro-advective effects %%
    % Check to ensure potential CC is not slightly lower than actual
    if NewCond.CC_NS < NewCond.CC
        NewCond.CC_NS = NewCond.CC;
        if tCC < Crop.CanopyDevEnd
            NewCond.CCxAct_NS = NewCond.CC_NS; 
        end
    end
    % Actual (with water stress)
    NewCond.CCadj = (1.72*NewCond.CC)-(NewCond.CC^2)+(0.3*(NewCond.CC^3)); 
    % Potential (without water stress)
    NewCond.CCadj_NS = (1.72*NewCond.CC_NS)-(NewCond.CC_NS^2)...
        +(0.3*(NewCond.CC_NS^3)); 
else
    % No canopy outside growing season - set various values to zero
    NewCond.CC = 0;
    NewCond.CCadj = 0;
    NewCond.CC_NS = 0;
    NewCond.CCadj_NS = 0;
    NewCond.CCxW = 0;
    NewCond.CCxAct = 0;
    NewCond.CCxW_NS = 0;
    NewCond.CCxAct_NS = 0;
end

end