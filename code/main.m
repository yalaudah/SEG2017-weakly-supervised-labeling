
clc; close all; clear;

% cd to the main directory before running the code

%% -------------------------------------------------------------------- %
%                              Main Parameters                          %
% --------------------------------------------------------------------- %

% Data Preperation ------------------------------------------------------
k = 100; % k in kmeans (i.e. number of features per class)
N_l = 4; % number of classes
data_skip = 1; % for 100 images only skip = 5. To use 500 images per class: skip = 1;
% Saving the results ----------------------------------------------------
save_skip = 1; % if you want to save only a subset of the images, use 2 or 5 or any other number
normalizeYfinal = 1; % if you'd like to normalize Yfinal to get probability values

%% -------------------------------------------------------------------- %
%                           Load and Prepare Data                       %
% --------------------------------------------------------------------- %

images = load('../data/images.mat');
images = images.images; % 2000 images of size 99x99
% skip images if you'd like (save memory)
images = images(1:data_skip:end, :,:);

numImages = size(images, 1);
% Assuming classes are balanced (each class has the same number of images) 
% -- if this is not the case, make sure you modify this: 
numImagesPerClass = numImages/N_l; 
X = reshape(images,numImages,99*99)'; % Data matrix

% make sure features are positive!
if min(X(:)) <0
    X = X-min(X(:));
    disp('Warning: Features are not all positive!');
end

% labels vector:
y = kron([1 2 3 4], ones(1,numImagesPerClass));
% apply kmeans on each set of labels to initialize W1:
X_ch = X(:, y==1); % Chaotic
X_ot = X(:, y==2); % Other
X_fa = X(:, y==3); % Fault
X_sa = X(:, y==4); % Salt Dome

% Use MATLAB's built in kmeans function:
[~, c_ch] = kmeans(X_ch',k, 'MaxIter',1000);
[~, c_ot] = kmeans(X_ot',k, 'MaxIter',1000);
[~, c_fa] = kmeans(X_fa',k, 'MaxIter',1000);
[~, c_sa] = kmeans(X_sa',k, 'MaxIter',1000);

% initialize W1_ W2_ and H_
W_init = [c_ch' c_ot' c_fa' c_sa'];
H_init = rand(N_l*k,numImages);

%% -------------------------------------------------------------------- %
%                                 Run ONMF                              %
% --------------------------------------------------------------------- %

% Sparsity Level: 
sW = 0.4; % Values from 0.4-0.85 work best

% create B matrix:
% B = eye(N_l*k,N_l*k);             % Identity matrix 
%  B = kron(eye(N_l,N_l),ones(k,k));% block diagonal 
B = rand(N_l*k,N_l*k);              % random matrix

% run ONMF:
save_memory = 1;
[W, H] = ONMF_SEG17(X,W_init,H_init,B,sW,N_l,save_memory);

H_t = squeeze(H(:,:,end));
W_t = squeeze(W(:,:,end));

% Q: binary cluster membership matrix
Q = kron(eye(N_l,N_l),ones(k,1));
Q = normalizeColumns(Q);
I_Nl = ones(1,N_l);
Y  = zeros(numImages,99*99,N_l); 

av_ch = zeros(N_l*k,1); hi_ch = zeros(N_l*k,1);
av_ot = zeros(N_l*k,1); hi_ot = zeros(N_l*k,1);
av_fa = zeros(N_l*k,1); hi_fa = zeros(N_l*k,1);
av_sa = zeros(N_l*k,1); hi_sa = zeros(N_l*k,1);

for img = 1:numImages
    clc;
    disp(['Image: ', num2str(img), '/' num2str(numImages)]);
    Hi = squeeze(H_t(:,img));
    
    switch y(img)
        case 1
            hi_ch = hi_ch + Hi; % not sorted
            av_ch = av_ch + N_l/numImages * sort(Hi,1,'descend'); %sorted
        case 2
            hi_ot = hi_ot + Hi; % not sorted
            av_ot = av_ot + N_l/numImages * sort(Hi,1,'descend');
        case 3
            hi_fa = hi_fa + Hi; % not sorted
            av_fa = av_fa + N_l/numImages * sort(Hi,1,'descend');
        case 4
            hi_sa = hi_sa + Hi; % not sorted
            av_sa = av_sa + N_l/numImages * sort(Hi,1,'descend');
    end
end

av = (av_ch+av_ot+av_fa+av_sa)/N_l;

% to find the cut-off point:
[res_x, idx_of_result] = knee_pt(av);
H_thresh = av(idx_of_result);

for img = 1:numImages
    clc;
    disp(['Image: ', num2str(img), '/' num2str(numImages)]);
    Hi = squeeze(H_t(:,img));
    Hi(Hi<H_thresh) = 0;
    Y(img,:,:) = W_t*(Q.*(Hi*I_Nl));
end

%% -------------------------------------------------------------------- %
%                               Save Results                            %
% --------------------------------------------------------------------- %

% create directory to save results:
t = datetime('now');
t.Format = 'ddMMMyyyy_hhmm';
dir_name = strcat('../results/res_',datestr(t,'mmddHHMM'));
mkdir(dir_name);
cd(dir_name);


idx = 0;
for i = 1:save_skip:numImages
    clc;
    disp([num2str(i),'/', num2str(numImages)]);
    img = reshape(squeeze(X(:,i)),[99 99]);
    Y_idx = squeeze(Y(i,:,:));
    
    if gaussian_filtering == 1
        for k = 1:N_l
            temp = reshape(squeeze(Y_idx(:,k)),[99,99]);
            temp = imgaussfilt(temp,sigma);
            Y_idx(:,k) = reshape(temp,[99*99,1]);
        end
    end
    
    [vals, classifiedImage] = max(Y_idx,[],2);
    conf = vals./(sum(Y_idx,2)+1e-6);
    classifiedImage =  reshape(classifiedImage,99,99);
    conf =  reshape(conf,99,99);
    classifiedImage(conf<conf_thresh) = 0;
    
    coloredImage =  uint8(zeros([size(img),3]));
    for ii = 1:size(img,1)
        for jj = 1:size(img,2)
            if classifiedImage(ii,jj) == 0     % Not Sure:
                coloredImage(ii,jj,:) = [183,183,183]; % Gray
            elseif classifiedImage(ii,jj) == 1     % Chaotic:
                coloredImage(ii,jj,:) = [0,0,255]; % blue
            elseif classifiedImage(ii,jj) == 2  % Other:
               coloredImage(ii,jj,:) = [183,183,183]; % default background
            elseif classifiedImage(ii,jj) == 3 % Fault:
                coloredImage(ii,jj,:) = [0,255,0]; % Green
            elseif classifiedImage(ii,jj) == 4 % Salt:
                coloredImage(ii,jj,:) = [255,0,0]; % red
            end
        end
    end
    
    close;
    hh = figure;
    imshow(img,[]);
    set(gca,'position',[0 0 1 1],'units','normalized'); % remove white border to save image only
    name = strcat('img_',num2str(i));
    saveas(hh, strcat(name,'.png'));

    hold on;
    h = imagesc(coloredImage);
    set(h, 'AlphaData', 0.45 );
    set(gca,'position',[0 0 1 1],'units','normalized'); % remove white border to save image only
    
    % save figure as an image with appropriate title:
    saveas(hh, strcat(name,'_labels.png'));
    
end