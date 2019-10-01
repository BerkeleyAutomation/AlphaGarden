function [] = AOS_WriteOutputs(AOS_ClockStruct,AOS_InitialiseStruct)
% Function to write output files 

%% Define output file location and name %%
FileLoc = AOS_InitialiseStruct.FileLocation.Output;
FileName = AOS_InitialiseStruct.FileLocation.OutputFilename;

%% Write outputs (new) %%
if AOS_ClockStruct.ModelTermination == true
    if strcmp(AOS_InitialiseStruct.FileLocation.WriteDaily,'Y')
        % Water contents
        fid = fopen(strcat(FileLoc,FileName,'_WaterContents.txt'),'a+t');
        fprintf(fid,strcat('%4.0f\t%2.0f\t%2.0f\t%8.0f\t%1.0f\t',...
            repmat('%5.4f\t',1,AOS_InitialiseStruct.Parameter.Soil.nComp),'\n'),...
            AOS_InitialiseStruct.Outputs.WaterContents');
        fclose(fid);
        % Water fluxes
        fid = fopen(strcat(FileLoc,FileName,'_WaterFluxes.txt'),'a+t');
        fprintf(fid,strcat('%4.0f\t%2.0f\t%2.0f\t%8.0f\t%1.0f\t%6.2f\t%6.2f\t%6.2f\t',...
            '%5.2f\t%5.2f\t%5.2f\t%5.2f\t%5.2f\t%5.2f\t%5.2f\t%5.2f\t%5.2f\t',...
            '%5.2f\t\n'),AOS_InitialiseStruct.Outputs.WaterFluxes');
        fclose(fid);
        % Crop growth
        fid = fopen(strcat(FileLoc,FileName,'_CropGrowth.txt'),'a+t');
        fprintf(fid,strcat('%4.0f\t%2.0f\t%2.0f\t%8.0f\t%1.0f\t%5.2f\t%6.2f\t%4.2f\t',...
            '%6.3f\t%6.3f\t%6.2f\t%6.2f\t%5.4f\t%5.4f\t%6.2f\t\n'),...
            AOS_InitialiseStruct.Outputs.CropGrowth');
        fclose(fid);
    end
    % Final output
    FinalOut = AOS_InitialiseStruct.Outputs.FinalOutput.';
    fid = fopen(strcat(FileLoc,FileName,'_FinalOutput.txt'),'a+t');
    fprintf(fid,'%4.0f\t%s\t%s\t%8.0f\t%s\t%8.0f\t%5.2f\t%6.2f\t\n',FinalOut{:});
    fclose(fid);
end

end