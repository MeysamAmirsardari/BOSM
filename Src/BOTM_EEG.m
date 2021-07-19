%                                               |
% EEG-Based Brain-operated typewriting machine  |
% M. Amirsardari - A. H. Mobasheri              |
% Summer 1400/2021                              |
% Part2: EEG Processing                         |
%_______________________________________________|

% Part2_EEG:
%Q4:
clear; clc; close all;

load SubjectData1.mat

time = train(1,:);
T = time(2)-time(1);
Fs = 1/T

%%
%Q6:
clc; close all;

for i = 2:9
    subplot(4,2,i-1);
    plotFFT(train(i,:), Fs)
    title(['Electrode ',num2str(i-1)]);
end

%%
% Q7_Energy spectrum:

clc; close all;

for i = 2:9
    subplot(4,2,i-1);
    spect = EnergyAccumulation(train(i,:));
    
    len = length(train(i,:));
    w = linspace(0, Fs/2, floor(len/2));

    plot(w, spect,'LineWidth',2);
    xlim([0, Fs/2])
    grid on
    grid minor
    title(['Energy Spectrum ',num2str(i-1)]);
end

%%
% Q8_Removing DC values:
clc; close all;

channels = train(2:9,:);
len = length(channels(1,:));

Row_means = mean(transpose(channels));

t = 1:1:len;
[DC, ~] = meshgrid(Row_means, t);
DC = transpose(DC);

channels = channels - DC;

for i = 1:8
    subplot(4,2,i);
    plotFFT(channels(i,:), Fs)
    title(['Electrode ',num2str(i),'without DC']);
end

%%
% Band Pass filtering:

close all; clc;
filteredChannels = zeros(8,len);

for i = 1:8
    filteredChannels(i,:) = bandpass(channels(i,:),[0.5, 40], Fs);
end

for i = 1:8
    subplot(4,2,i);
    plotFFT(filteredChannels(i,:), Fs)
    title(['Filtered Channel',num2str(i)]);
end

%%
% Q9_Down Sampling:
clc; close all;

L = 3;
downSampled = zeros(8,floor(len/L));

temp1 = [train(1,:); filteredChannels; train(10:11,:)];

for i = 1:11
    downSampled(i,:) = downSampler(temp1(i,:), L);
end

figure
for i = 1:8
    subplot(4,2,i);
    plotFFT(downSampled(i+1,:), Fs)
    title(['Filtered Channel',num2str(i)]);
end

%%
% Q10_epoching:
clc; close all;

N = length(downSampled(10,:));
StimuliOnset = [];

for i = 1:N-1
    if((downSampled(10,i)==0)&&(downSampled(10,i+1)~=0))         
        StimuliOnset = [StimuliOnset, i+1];
    end   
end

epoched = epoching(downSampled, 0.2, 0.8, StimuliOnset);

%%
save('epochedData1.mat','epoched');

%%
% Q12:
clc; close all;
Fs2 = 3/(downSampled(1,6)-downSampled(1,3));

L = length(epoched(1,:,1));
filter100 = fir1(84,[20/Fs2, 80/Fs2], kaiser(85,3));

sig = zeros(8, L, length(epoched(1,1,:)));

for i = 1:8
    for k = 1:L
      sig(i,k,:) = filter(filter100, 1, epoched(i,k,:));
    end
end


%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Functions:

function plotFFT(InputSignal, Fs)
    transformed = fft(InputSignal);
    len = length(InputSignal);
       
    spect = abs(transformed(1:floor(len/2)))/len;
    w = linspace(0, Fs/2, floor(len/2));
    
    plot(w, spect,'LineWidth',2);
    title('FFT output');
    xlim([0, Fs/2])
    grid on
    grid minor
end


function E = EnergyAccumulation(signal)
    syms n;
    
    transformed = fft(signal);
    len = length(signal);
    N = floor(len/2);
    
    spectrum = abs(transformed(1:N))/len;
    Energy = zeros(N,1);
    
    for i = 1:N
        if (i>200)
            %spectrum(i) = symsum((signal(n))^2, n, 50, i);
            sums = 0;
            for k = 200:i
                sums = sums + (spectrum(k))^2;
            end
            Energy(i) = sums;
        end
    end

    E = Energy;
end

function out = downSampler(Signal, L)
    len = length(Signal);
    N = floor(len/L);
    out = zeros(N,1);
    
    for i=1:N
             out(i) = Signal(L*i);  
    end
end


function epoched = epoching(InputSignal,BackwardSamples,ForwardSamples,StimuliOnset)
    Fs = 3/(InputSignal(1,6)-InputSignal(1,3));
    
    BckIdx = floor(BackwardSamples*Fs);
    ForIdx = floor(ForwardSamples*Fs);
    N = length(StimuliOnset);
    
    epoched = zeros(8, N, BckIdx+ForIdx);
    
    for i = 1:N
        startIdx = StimuliOnset(i)-BckIdx;
        endIdx = StimuliOnset(i)+ForIdx-1;
        
        epoched(:,i,:) = InputSignal(2:9, startIdx:endIdx);  
    end
end





