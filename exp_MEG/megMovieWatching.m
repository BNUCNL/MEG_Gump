function megMovieWatching(subjectID,runID)
% megMovieWatching(subjID,runID)
% subjID, subject id, string
% runID, run id, integer
% Subject presses 3 or 4 key with right hand indicate she/he is ready.
% Then, the scanner or experimenter presses S key to begin the experiment.
% By Xingyu Liu && Zonglei Zhen @ BNU,2019.04

%% initializing
% Screen('Preference', 'SkipSyncTests', 1);
HideCursor;
ListenChar(2);
AssertOpenGL;

%IO setting
ioObj = io64;
status = io64(ioObj);
address = hex2dec('D020');
data_out=0;
io64(ioObj,address,data_out);

%% Parameter setting
% Keys setting
KbName('UnifyKeyNames');
escKey = KbName('ESCAPE');
readyKey = [KbName('3#'),KbName('4$')];
sKey = KbName('s');

% marker setting
initialMarker = 255;
markerDur = 0.005;

% file path setting
moviePath  = fullfile(pwd,'materials','movie_1024x768_15min');
resultPath = fullfile(pwd,'results','watching_results',subjectID);
if exist(resultPath,'dir') ~= 7
    mkdir(resultPath);
end
date = strrep(strrep(datestr(clock),':','-'),' ','-');
outFileName = fullfile(resultPath,sprintf('%s_fg_av_chi_seg_%d_%s.mat',subjectID,runID,date));

% instruciton path setting
watchingInsPath = fullfile(pwd,'materials','ins_watching.jpg');
fixPath = fullfile(pwd,'materials','fixation.jpg');

disp('======setting done=====')

%% Open window
screenNum = max(Screen('Screens'));
[wptr,wrect] = Screen('OpenWindow', screenNum, 0);

%% Load movie
movieName = fullfile(moviePath,['fg_av_seg', num2str(runID),'_chi.mp4']);
[movie,~,~,~,~,framecount] = Screen('OpenMovie',wptr,movieName,4,-1);

%% Load instruciton
watchingIns = Screen('MakeTexture',wptr,imread(watchingInsPath));
fixIns = Screen('MakeTexture',wptr,imread(fixPath));


%% show start instruciton
Screen('DrawTexture',wptr,watchingIns,[],wrect);
Screen('Flip',wptr);

% Wait subject to be ready
while KbCheck(); end
while true
    [keyIsDown, ~, keyCode] = KbCheck();
    if keyIsDown && sum(keyCode(readyKey))
        break;
    elseif keyIsDown && keyCode(escKey)
        scaReset
        return
    end
end

%% wait for meg start & set initial marker to trigger MEG
Screen('DrawTexture', wptr, fixIns,[],wrect);
Screen('Flip', wptr);

while KbCheck(); end
while true
    [keyIsDown, tKey, keyCode] = KbCheck();
    if keyIsDown && keyCode(sKey)
        % Send exp start mark
        io64(ioObj,address,initialMarker);
        expStartTime = tKey;
        while GetSecs()-expStartTime < markerDur, end
        io64(ioObj,address,0);
        break;
    elseif keyIsDown && keyCode(escKey)
        scaReset
        return
    end
end

%% Play move until end
index = 1;
framesTime = zeros(framecount,2);
Screen('PlayMovie', movie, 1);

while KbCheck(); end
while true
    [keyIsDown,~,keyCode] = KbCheck();
    % Wait for next movie frame, retrieve texture handle to it
    [tex,pts] = Screen('GetMovieImage', wptr, movie);
    framesTime(index,1) = pts;
    
    %  0 or -1 indicate movie has reached movie end
    if tex <= 0, break;  end
    
    % Show frame
    Screen('DrawTexture', wptr, tex,[],wrect);
    frameShowTime = Screen('Flip', wptr);
    
    % Set Maker here
    io64(ioObj,address, ceil(pts/10));
    while GetSecs()-frameShowTime < markerDur, end
    io64(ioObj,address,0);
    
    Screen('Close', tex)
    framesTime(index,2) = frameShowTime - expStartTime;
    index = index+1;
    
    % Breaking the movie when esc is pressed
    if keyIsDown && keyCode(escKey)
        save(outFileName);
        fprintf('Data were saved to: %s\n',outFileName);
        scaReset
        disp('ESC is pressed to abort the program.');
        
        return;
    end
end

% Stop playback
Screen('PlayMovie', movie, 0);

% Close movie
Screen('CloseMovie', movie);

%% save data
fprintf('Data were saved to: %s\n',outFileName);
save(outFileName);

scaReset


function scaReset
Screen('CloseAll');
ShowCursor;
ListenChar(0);