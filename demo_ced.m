% demonstration of color edge detector
% ANDD matrices construction
function demo_ced
addpath('D:\demo\color edge dataset\');
trainroot = 'D:\demo\color edge dataset\train\images\';
gtroot = 'D:\demo\color edge dataset\train\gt\';
imtrain = dir([trainroot '*.tif']);
load('D:\demo\settings.mat','paras');
X = zeros(length(paras),30000*30);
Y = zeros(1,90000*30);
m = 1;
for n = 1:30
    img = imread([trainroot imtrain(n).name]);
    CESM = CESM_ANDD(img,paras);
    gts = load([gtroot imtrain(n).name(1:end-4) '.mat']);
    gt = gts.gt(:)';
    t=(gt==0);
    num = numel(t)-1;
    X(:,m:m+num) = CESM;
    Y(1,m:m+num)=t;
    m = m+num;
    clear CESM  t
end
X(:,m:end)=[];
Y(:,m:end)=[];
% Create a Pattern Recognition Network
hiddenLayerSize = 10;
net = patternnet(hiddenLayerSize);
net.divideParam.trainRatio = 80/100;
net.divideParam.valRatio = 20/100;
net.divideParam.testRatio = 0/100;
% Train the Network
[net,~] = train(net,X,Y);
save('D:\demo\edge_net','net')
% vote for angle
load('D:\demo\edge_net.mat','net');
% load('D:\demo\edge_net.mat','net');
testroot = 'D:\demo\color edge dataset\test\images\';
imtest = dir([testroot '*.tif']);
results = cell(length(imtest),1);
% for i2 = 1:length(imtest)
for i2 = 1:1
    img2 = imread([testroot imtest(i2).name]);
    [CESM,CEDM] = CESM_ANDD(img2,paras);
    EPM = net(CESM);
    EDM = select_angle(CEDM);
    results{i2,1}.epm = reshape(EPM(1,:),size(img2,1),size(img2,2));
    results{i2,1}.edm = reshape(EDM,size(img2,1),size(img2,2));
    % non-maximum suppresion and hysteresis thresholding
    magS = sort(EPM(:));
    [m,n] = size(results{i2,1}.epm);
    lowThresh = magS(floor(0.8*m*n));
    highThresh = magS(floor(0.9*m*n));
    if lowThresh >= highThresh
        continue;
    end
    idxStrong = [];
    e =false(m,n);
    for dirs = 1:4
        idxLocalMax = mycannyFindLocalMaxima(dirs,results{i2,1}.edm,results{i2,1}.epm);
        idxWeak = idxLocalMax(results{i2,1}.epm(idxLocalMax) > lowThresh);
        e(idxWeak)=1;
        idxStrong = [idxStrong; idxWeak(results{i2,1}.epm(idxWeak) > highThresh)]; %#ok<AGROW>
    end
    if ~isempty(idxStrong) % result is all zeros if idxStrong is empty
        rstrong = rem(idxStrong-1, m)+1;
        cstrong = floor((idxStrong-1)/m)+1;
        e = bwselect(e, cstrong, rstrong, 8);
    end
    % output binary edge detection result
    e(1:5,:)=0;e(end-4:end,:)=0;e(:,1:5)=0;e(:,end-4:end)=0;
    results{i2,1}.bem = e;
    results{i2,1}.ths =[lowr highr];
end
save results_1v3.mat results
function [esm,edm] = CESM_ANDD(img,paras,inds)
% inds : index of positive and negative samples
% positive samples: CESMs of all edge pixels
% negative samples: CESMs of sampled non-edge pixels
% where the numbers of two types samples are equal

P     =16;    % number of directions
wd = 10;     % size of ANDD filters
img = double(img);
gau = fspecial('Gaussian',round(1.5)*2+1,1);
img=imfilter(img,gau','conv','same');
[row,col] = size(img(:,:,1));
if exist('inds','var')
    inds = sort(inds);
    esm = zeros(225,length(inds));
    edm = zeros(225,length(inds));
    scales = paras(:,1); afs = paras(:,2);
    parfor i = 1:length(paras)
        %--------------------------------------------------------------------
        % Construct the ANDD filters
        BB=anisotropic_Directional_derivative_filter(scales(i),P,afs(i),wd);
        % calculate ANDD responses
        tempConv = zeros(size(img,1),size(img,2),3,P);
        for j=1:P
            tempConv(:,:,:,j) = imfilter(img,BB(:,:,j),'symmetric');
        end
        %%
        RGB = [reshape(tempConv(:,:,1,:),[],16);reshape(tempConv(:,:,2,:),[],16);reshape(tempConv(:,:,3,:),[],16)];
        esm1 = zeros(length(inds),1);
        num = row*col;
        for n = 1:length(inds)
            J2 = RGB([inds(n) inds(n)+num inds(n)+num*2],:);
            [~,s2,~]=svd(J2,'econ');
            esm1(n)=s2(1);
        end
        esm(i,:) =esm1;
    end
else
    esm = zeros(225,row*col);
    edm = zeros(225,row*col);
    scales = paras(:,1); afs = paras(:,2);
    %     for i = 1:length(paras)
    parfor i = 1:length(paras)
        %--------------------------------------------------------------------
        % Construct the ANDD filters
        BB=anisotropic_Directional_derivative_filter(scales(i),P,afs(i),wd);
        % calculate ANDD responses
        tempConv = zeros(size(img,1),size(img,2),3,P);
        for j=1:P
            tempConv(:,:,:,j) = imfilter(img,BB(:,:,j),'symmetric');
        end
        %%
        RGB = [reshape(tempConv(:,:,1,:),[],16);reshape(tempConv(:,:,2,:),[],16);reshape(tempConv(:,:,3,:),[],16)];
        esm1 = zeros(row*col,1);
        num = numel(esm1);
        v2 = zeros(16,3,num);
        for n = 1:num
            J2 = RGB([n n+num n+num*2],:);
            [~,s2,v2(:,:,n)]=svd(J2,'econ');
            esm1(n)=s2(1);
        end
        [~,id2]= max(abs(squeeze(v2(:,1,:))));
        id2= mod(id2-1,P)+1;
        edm(i,:) = (id2-1)*pi/P-pi/2;
        esm(i,:) =esm1;
    end
end
function BB=anisotropic_Directional_derivative_filter(cgma,P,rou,wd)
% calculate angk kernel
cta=0:pi/P:pi-pi/P;
x=-wd:1:wd;
y=-wd:1:wd;
BB = zeros(2*wd+1,2*wd+1,P);
for k=1:P
    B=kernel_matrix(rou,cta(k));
    for m=1:2*wd+1
        for n=1:2*wd+1
            z=[x(m);y(n)];
            BB(m,n,k)=(-rou*(x(m)*cos(cta(k))+y(n)*sin(cta(k)))/cgma)*1/(2*pi*cgma)*exp(-z'*B*z/(2*cgma));
        end
    end
end
function  AA=kernel_matrix(rou,cta)
AA=zeros(2,2);
AA(1,1)=rou*cos(cta)^2+sin(cta)^2/rou;
AA(1,2)=(rou-1/rou)*cos(cta)*sin(cta);
AA(2,1)=AA(1,2);
AA(2,2)=rou*sin(cta)^2+cos(cta)^2/rou;
% select the angle by  vote rule
% if two or more angles are voted, use the averaged result
function A = select_angle(x)
[rows,cols] = size(x);
x = sort(x,1);
xn = [ones(1,cols); x(1:end-1,:)~=x(2:end,:)];
xn = find(xn(:));
freq = zeros([rows,cols],'like',full(double(x([]))));
freq(xn) = [xn(2:end); numel(x)+1]-xn;
[maxfreq,~] = max(freq,[],1);
A = zeros(cols,1);
selection = freq == maxfreq;
for j = 1:cols
    A(j) = mean(x(selection(:,j),j));
end

function idxLocalMax = mycannyFindLocalMaxima(direction,grad_dir,mag)

% This sub-function helps with the non-maximum suppression in the Canny
% edge detector.  The input parameters are:
%
%   direction - the index of which direction the gradient is pointing,
%               read from the diagram below. direction is 1, 2, 3, or 4.
%   ix        - input image filtered by derivative of gaussian along x
%   iy        - input image filtered by derivative of gaussian along y
%   mag       - the gradient magnitude image
%
%    there are 4 cases:
%
%                         The X marks the pixel in question, and each
%         3     2         of the quadrants for the gradient vector
%       O----0----0       fall into two cases, divided by the 45
%     4 |         | 1     degree line.  In one case the gradient
%       |         |       vector is more horizontal, and in the other
%       O    X    O       it is more vertical.  There are eight
%       |         |       divisions, but for the non-maximum suppression
%    (1)|         |(4)    we are only worried about 4 of them since we
%       O----O----O       use symmetric points about the center pixel.
%        (2)   (3)


[m,n] = size(mag);

% Find the indices of all points whose gradient (specified by the
% vector (ix,iy)) is going in the direction we're looking at.

switch direction
    case 1
        idx = find(grad_dir>=0 & grad_dir<=pi/4);
    case 2
        idx = find(grad_dir>pi/4 & grad_dir<=pi/2);
    case 3
        idx = find(grad_dir>=-pi/2 & grad_dir<=-pi/4 );
    case 4
        idx = find(grad_dir>-pi/4 & grad_dir<0);
end

% Exclude the exterior pixels
if ~isempty(idx)
    v = mod(idx,m);
    extIdx = (v==1 | v==0 | idx<=m | (idx>(n-1)*m));
    idx(extIdx) = [];
end

% ixv = ix(idx);
% iyv = iy(idx);
gradmag = mag(idx);

% Do the linear interpolations for the interior pixels
switch direction
    case 1
        d = tan(grad_dir(idx));
        gradmag1 = mag(idx+m).*(1-d) + mag(idx+m-1).*d;
        gradmag2 = mag(idx-m).*(1-d) + mag(idx-m+1).*d;
    case 2
        d = tan(pi/2-grad_dir(idx));
        gradmag1 = mag(idx-1).*(1-d) + mag(idx+m-1).*d;
        gradmag2 = mag(idx+1).*(1-d) + mag(idx-m+1).*d;
    case 3
        d = tan(pi/2+grad_dir(idx));
        gradmag1 = mag(idx-1).*(1-d) + mag(idx-m-1).*d;
        gradmag2 = mag(idx+1).*(1-d) + mag(idx+m+1).*d;
    case 4
        d = -tan(grad_dir(idx));
        gradmag1 = mag(idx-m).*(1-d) + mag(idx-m-1).*d;
        gradmag2 = mag(idx+m).*(1-d) + mag(idx+m+1).*d;
end
idxLocalMax = idx(gradmag>=gradmag1 & gradmag>=gradmag2);