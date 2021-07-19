%                                               |
% EEG-Based Brain-operated typewriting machine  |
% M. Amirsardari - A. H. Mobasheri              |
% Summer 1400/2021                              |
% Part4: Filtering                              |
%_______________________________________________|

% Filter Design

clear 
close all
clc
x = 0:1:99;
y = cos(x);
%y = sinc(x/17)
figure
plot(x,angle(y))
gd = groupdelay(y,100);
[gdo,wo] = grpdelay(y,1,100);
figure
plot(gd)
figure 
plot(gdo)
gd

%%% ----- Functions---------
function out = groupdelay(h,N)

    n = linspace(0,N - 1 ,N);
    h = [h zeros(1,N - length(h))];
    nh = n.* h;
    jdH = fft(nh,N);
    H = fft(h,N);
    out = real(jdH ./ H);
    out1 = out;
    out1(isinf(out1)) = [];
    out1(isnan(out1)) = [];
  
    mea = mean(out1);
    out(isinf(out)) = mea;
    out(isnan(out)) = mea;
   
end

function out = zphasefilter(h,x)

filtered = conv(h,x); %1d convolution for filtering
gd = groupdelay(h,lenght(h) + 50);

delay = floor(mean(gd));

if ( delay > 0 )
        unshifted = filtered(1+delay:end);
        filtered(1:end-delay) = unshifted;    
elseif (delay < 0 )
    
        unshifted = filtered(1:end-abs(delay));
        filtered(abs(delay):end) = unshifted; 
    
end

out = filtered;


end