%                                               |
% EEG-Based Brain-operated typewriting machine  |
% M. Amirsardari - A. H. Mobasheri              |
% Summer 1400/2021                              |
% Part1: Sampling                               |
%_______________________________________________|

% Part1_ Sampling:
% A:
clear; clc; close all;

Fs = 1000;
w = 0:1/Fs:2*pi-1/Fs;

triPulse = triangularPulse(w/pi-1);
%triPulse = (w.*heaviside(w) + (2*pi-2*w).*heaviside(w-pi)).*heaviside(2*pi-w)/pi;

plot(w, triPulse, 'LineWidth',2)
title('Frequency Domain input')
xlim([0, 8])
grid on
grid minor

InputSignal = ifft(triPulse*length(triPulse));

HalfBandFFT(InputSignal, Fs);

%%
% C:
clear; clc; close all;

Fs = 1000;
syms n;
w = 0:1/Fs:12*pi-1/Fs;

triPulse = @(w) ((w/6).*heaviside(w/6)+(2*pi-w/3).*heaviside(w/6-pi)).*heaviside(2*pi-w/6)/pi;

%sig2 = symsum(triPulse(w-2*pi*n),n,-5,5);

sig2 = triPulse(w-10*pi)+triPulse(w-8*pi)+triPulse(w-6*pi)+...
        +triPulse(w-4*pi)+triPulse(w-2*pi)+...
        +triPulse(w)+triPulse(w+2*pi)+triPulse(w+4*pi)+...
        +triPulse(w+6*pi)+triPulse(w+8*pi)+triPulse(w+10*pi);

hold on
plot(w, sig2, 'LineWidth',2)
plot(w, triPulse(w), 'LineWidth',2)
legend('Periodic pulse with aliasing','Single pulse')
title('Down sample by 6')
grid on
grid minor
hold off

%%
clear; clc; close all;

Fs = 1000;
syms n;
w = 0:1/Fs:8*pi-1/Fs;
t = -30:1/Fs:30-1/Fs;

sig1 = 0.5*(sawtooth(w+pi,1/2)+1);

triPulse = @(w) ((w/4).*heaviside(w/4)+(2*pi-w/2).*heaviside(w/4-pi)).*heaviside(2*pi-w/4)/pi;

%%sig2 = symsum(triPulse(w-2*pi*n),n,-3,3);
sig2 = triPulse(w-6*pi)+triPulse(w-4*pi)+triPulse(w-2*pi)+...
        +triPulse(w)+triPulse(w+2*pi)+triPulse(w+4*pi)+...
        +triPulse(w+6*pi);

hold on
plot(w, sig2, 'LineWidth',2)
plot(w, triPulse(w), 'LineWidth',2)
legend('Periodic pulse with aliasing','Single pulse')
title('Down sample by 4')
grid on
grid minor
hold off

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%
% Functions:

function HalfBandFFT(InputSignal, Fs)
    transformed = fftshift(fft(InputSignal));
    len = length(InputSignal);
    abs_trans = abs(transformed(1:floor(len/2)));
    
    w = linspace(0, pi, floor(len/2))*Fs;
    
    
    spect = abs(transformed)/len;
    w2 = linspace(0, 2*pi*Fs, len);
    
    figure
    plot(w2, spect,'LineWidth',2);
    title('FFT output');
    xlim([0, 4*Fs])
    grid on
    grid minor

    figure
    plot(w, abs_trans,'LineWidth',2);
    title('HalfBandFFT output');
    xlim([0, 4*Fs])
    grid on
    grid minor
end












