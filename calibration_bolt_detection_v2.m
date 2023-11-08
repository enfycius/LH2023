image_list = dir('./*.jpeg')

disp(length(image_list))

imageFileNames = {};

for i = 1:length(image_list)
   filename = image_list(i).name;

   imageFileNames = [imageFileNames, filename];
end

% disp(strcat(images.folder, images.name))

% disp(images.name)

% imageFileNames = imread(images.name);

[imagePoints, boardSize] = detectCheckerboardPoints(imageFileNames);

% images.name
%%
squareSizeInMM = 29;
worldPoints = generateCheckerboardPoints(boardSize, squareSizeInMM);
%% 
I = imread("./../calibration/saved_pypylon_img_zlns.jpeg");
imageSize = [size(I, 1), size(I, 2)];
params = estimateCameraParameters(imagePoints, worldPoints, ...
    'ImageSize', imageSize);

%% 
figure;
imshow(imageFileNames{1});
hold on;
plot(imagePoints(:, 1, 1), imagePoints(:, 2, 1), 'go');
plot(params.ReprojectedPoints(:, 1, 1), params.ReprojectedPoints(:, 2, 1), 'r+');
legend('Detected Points', 'ReprojectedPoints');

hold off;

%% 
I = imread("./../calibration/saved_pypylon_img_zlns.jpeg");
J1 = undistortImage(I, params);
figure; imshowpair(I, J1, 'montage');
title('Original Image (left) vs. Corrected Image (right)');

imwrite(J1, "./../calibration/result.jpeg");

%%
%% Edge Detection Using Fuzzy Logic

% Create Fuzzy Inference System (FIS)
edgeFIS = mamfis('Name', 'edgeDetection');

% Specify inputs: image gradients are used as inputs
edgeFIS = addInput(edgeFIS, [-3.5,3.5], 'Name', 'Ix');
edgeFIS = addInput(edgeFIS, [-3.5,3.5], 'Name', 'Iy');

% Specify membership function for each input
sx = 0.1; % std
mux = 0; % mean
sy = 0.1; % std
muy = 0; % mean
edgeFIS = addMF(edgeFIS, 'Ix', 'gaussmf', [sx mux], 'Name', 'zero');
edgeFIS = addMF(edgeFIS, 'Iy', 'gaussmf', [sy muy], 'Name', 'zero');

% Specify output
edgeFIS = addOutput(edgeFIS, [0,1], 'Name', 'Iout');

% Specify membership function for output
% Original
whiteval = [0.1, 1, 1];
blackval = [0, 0, 0.7];
% Switch black and white
% whiteval = [0.3, 1, 1];
% blackval = [0, 0, 0.9];

edgeFIS = addMF(edgeFIS, 'Iout', 'trimf', whiteval, 'Name', 'edge');
edgeFIS = addMF(edgeFIS, 'Iout', 'trimf', blackval, 'Name', 'non-edge');

% figure('position', [50 50 1100 400]), 



% t1.TileSpacing = 'compact';
% t1.Padding = 'compact';
% 
% figure, 
% subplot(1,2,1)
% plotmf(edgeFIS, 'input', 1)
% set(gca,'FontName','times','FontSize',12,'FontWeight','bold')
% 
% subplot(1,2,2)
% plotmf(edgeFIS, 'output', 1)
% set(gca,'FontName','times','FontSize',12,'FontWeight','bold')

% figure('position', [50 50 800 800]), 
tr = tiledlayout(1,2);

nexttile
plotmf(edgeFIS, 'input', 1)
set(gca,'FontName','times','FontSize',12,'FontWeight','bold')

nexttile
plotmf(edgeFIS, 'output', 1)
set(gca,'FontName','times','FontSize',12,'FontWeight','bold')

tr.TileSpacing = 'compact';
tr.Padding = 'compact';

% title('Iout')

% Specify Rules
% r1 = "If Ix is zero and Iy is zero then Iout is white";
% r2 = "If Ix is not zero or Iy is not zero then Iout is black";
% r1 = "Ix==zero & Iy==zero => Iout=white";
% r2 = "Ix~=zero | Iy~=zero => Iout=black";
r1 = "Ix==zero & Iy==zero => Iout=non-edge";
r2 = "Ix~=zero | Iy~=zero => Iout=edge";
edgeFIS = addRule(edgeFIS, [r1, r2]);
edgeFIS.Rules

%% Load Image




% figure('position', [50 50 1100 400]), 
% 
% t1 = tiledlayout(1,2);
% 
% nexttile
% image(Irgb, 'CDataMapping', 'scaled')
% colormap('jet')
% title('Input Image in RGB')
% axis equal
% axis tight
% 
% nexttile
% image(Igray, 'CDataMapping', 'scaled')
% colormap('gray')
% title('Input Image in Grayscale')
% axis equal
% axis tight
% 
% t1.TileSpacing = 'compact';
% t1.Padding = 'compact';


% figure, 
% t1 = tiledlayout(1,1);
% nexttile
% image(Irgb, 'CDataMapping', 'scaled')
% colormap('jet')
% % title('Input Image in RGB')
% axis equal
% axis tight
% t1.TileSpacing = 'compact';
% t1.Padding = 'compact';
% 
% figure, 
% t1 = tiledlayout(1,1);
% nexttile
% image(Igray, 'CDataMapping', 'scaled')
% colormap('gray')
% % title('Input Image in Grayscale')
% axis equal
% axis tight
% t1.TileSpacing = 'compact';
% t1.Padding = 'compact';



%% Preprocessing - modified
input_folder = './../calibration/after/';
imagefiles=dir(fullfile(input_folder,'*.jpeg'));
nfiles = length(imagefiles);

%h = waitbar(0,' Time Loop: Fuzzy based edge detection');

 for i=1:nfiles
     i
   currentfilename = fullfile(input_folder,imagefiles(i).name);
   currentimage = imread(currentfilename);
   Irgb = currentimage;

    % Convert to Grayscale from RGB
    Igray = rgb2gray(Irgb);

    GF = sqrt(16);

    % Convert to double-precision data
    I = im2double(Igray);

    % Gaussian filter
    I = imgaussfilt(I, GF);

    % Obtain Image Gradient
    [Ix, Iy] = imgradientxy(I,'sobel');
    % Other filters can be used to obtain image gradients
    % Such as: Sobel, Prewitt
    % Functions: imfilter, imgradientxy, imgradient
    % Evaluate FIS
    Ieval = zeros(size(I));
    for ii = 1:size(I,1)
        Ieval(ii,:) = evalfis(edgeFIS, [(Ix(ii,:));(Iy(ii,:))]');
    end
    save_folder ='./../calibration/before/results/';
    save_filename = fullfile(save_folder,strcat(imagefiles(i).name(1:end-4),'png'));
    imwrite(Ieval,save_filename);
    
%    waitbar(i/nfiles,h);
 end

figure;
imshow(Ieval);
title('Original Image');% 원의 반지름 범위 설정
radiiRange = [10 30]; % 예를 들어, 20에서 50 픽셀 사이의 반지름을 가진 원을 찾습니다.

% 원 찾기
[centers, radii] = imfindcircles(Ieval, radiiRange, 'ObjectPolarity','bright', 'Sensitivity',0.60);

% 찾은 원 표시
viscircles(centers, radii,'EdgeColor','b');
hold on;plot(centers(:,1), centers(:,2), 'r.', 'MarkerSize', 10);
title('Detected Circles');

[imgHeight, imgWidth, ~] = size(Ieval);

distToLeftEdge = centers(:, 1);
distToRightEdge = imgWidth - centers(:, 1);
distToTopEdge = centers(:, 2);
distToBottomEdge = imgHeight - centers(:, 2);

% 각 모서리에서 가장 가까운 점 찾기
[~, leftIdx] = min(distToLeftEdge);
[~, rightIdx] = min(distToRightEdge);
[~, topIdx] = min(distToTopEdge);
[~, bottomIdx] = min(distToBottomEdge);

closestToLeft = centers(leftIdx, :);
closestToRight = centers(rightIdx, :);
closestToTop = centers(topIdx, :);
closestToBottom = centers(bottomIdx, :);


hold on;plot(closestToLeft(1), closestToLeft(2), 'c.', 'MarkerSize', 10);
hold on;plot(closestToRight(1), closestToRight(2), 'c.', 'MarkerSize', 10);
hold on;plot(closestToTop(1), closestToTop(2), 'c.', 'MarkerSize', 10);
hold on;plot(closestToBottom(1), closestToBottom(2), 'c.', 'MarkerSize', 10);


% 원본 이미지 좌표 (영상의 가장자리에서 가장 가까운 centers 점들)
movingPoints = [closestToLeft; closestToTop; closestToRight; closestToBottom];
bottomPoints = [closestToBottom;closestToRight];
deltaY = diff(bottomPoints(:,2));
deltaX = diff(bottomPoints(:,1));
angleRad = atan2(deltaY, deltaX);
angleDeg = rad2deg(angleRad);
angleDeg = -angleDeg;
% 3. 이미지와 centers 회전
rotatedImg = imrotate(Ieval, -angleDeg, 'bilinear', 'crop');
rotatedCenters = (centers - mean(centers)) * [cos(-angleRad) -sin(-angleRad); sin(-angleRad) cos(-angleRad)] + mean(centers);

% 결과 확인
figure;
% 원 찾기
[centers, radii] = imfindcircles(rotatedImg, radiiRange, 'ObjectPolarity','bright', 'Sensitivity',0.60);

% 찾은 원 표시
hold on;plot(centers(:,1), centers(:,2), 'r.', 'MarkerSize', 20);
title('Detected Circles');

% figure, 
% t1 = tiledlayout(1,1);
% nexttile
% image(Ix, 'CDataMapping', 'scaled')
% colormap('gray')
% title('Ix','FontName','times','FontSize',14,'FontWeight','bold')
% axis equal
% axis tight
% % set(gca,'FontName','times','FontSize',14,'FontWeight','bold')
% t1.TileSpacing = 'compact';
% t1.Padding = 'compact';
% 
% figure, 
% t1 = tiledlayout(1,1);
% nexttile
% image(Iy, 'CDataMapping', 'scaled')
% colormap('gray')
% title('Iy','FontName','times','FontSize',14,'FontWeight','bold')
% axis equal
% axis tight
% % set(gca,'FontName','times','FontSize',14,'FontWeight','bold')
% t1.TileSpacing = 'compact';
% t1.Padding = 'compact';
% 
% 
% BW1 = edge(Igray, 'Canny', [], 0.5);
% % BW1 = edge(Igray,'Canny', [], sqrt(2));
% figure,
% image(BW1, 'CDataMapping', 'scaled')
% colormap('gray')







% % Plot detected edges
% figure, 
% t1 = tiledlayout(1,1);
% nexttile
% image(Ieval, 'CDataMapping', 'scaled')
% colormap('gray')
% % title('Edge Detection Using Fuzzy Logic')
% axis equal
% axis tight
% % set(gca,'FontName','times','FontSize',14,'FontWeight','bold')
% t1.TileSpacing = 'compact';
% t1.Padding = 'compact';

%%
figure;
imshow(Ieval);
impixelinfo;

%%
imwrite(Ieval, "./test.png")

%%
I = Ieval;

% initial control points
% A = [336 250;  3285 415; 3270 679; 318 508];
% B = [0 0; 5577 0;  5577 540;  0 540];

input=[closestToTop; closestToRight; closestToBottom; closestToLeft];
inpX=input(:,1);
inpY=input(:,2);
% inpX = [336;3285;3270;318];
% inpY = [250;415;679;508];
OutX = [0;2952;2952;0];
OutY = [0;0;264;264];

T = fitgeotrans([inpX inpY],[OutX OutY],'projective');
J = imwarp(I, T);
%%
% transform input image and show result
figure(2);
subplot(121), imshow(I), title('image');
subplot(122), imshow(J), title('warped');

%%
figure;
imshow(Ieval);

%%
figure;
imshow(J);
%%
figure;
imshow(J);
impixelinfo;
%%
figure;
J2 = imcrop(J,[150 320 3525 875]);
imshow(J2);

%%
radiiRange = [6 100]; % 예를 들어, 20에서 50 픽셀 사이의 반지름을 가진 원을 찾습니다.

% 원 찾기
[centers, radii] = imfindcircles(Ieval, radiiRange, 'ObjectPolarity','bright', 'Sensitivity',0.60);
%%
warping = imread('./warping.png');

figure;
imshow(warping);

%%
warping2 = imread('./warping2.png');

figure;
imshow(warping2);

%% Preprocessing - modified
figure;

radiiRange = [5 2000]; % 예를 들어, 20에서 50 픽셀 사이의 반지름을 가진 원을 찾습니다.

% 원 찾기
[centers, radii] = imfindcircles(warping2, radiiRange, 'ObjectPolarity','bright', 'Sensitivity',0.75);

hold on;

% imshow(warping2);
% 찾은 원 표시
% viscircles(centers, radii,'EdgeColor','b');

hold on;plot(centers(:,1), centers(:,2), 'r.', 'MarkerSize', 10);
title('Detected Circles');
% % % 
yticks(0:2.5:800);
% % % 
grid on;

% [imgHeight, imgWidth, ~] = size(Ieval);
% 
% distToLeftEdge = centers(:, 1);
% distToRightEdge = imgWidth - centers(:, 1);
% distToTopEdge = centers(:, 2);
% distToBottomEdge = imgHeight - centers(:, 2);

% 각 모서리에서 가장 가까운 점 찾기
% [~, leftIdx] = min(distToLeftEdge);
% [~, rightIdx] = min(distToRightEdge);
% [~, topIdx] = min(distToTopEdge);
% [~, bottomIdx] = min(distToBottomEdge);
% 
% closestToLeft = centers(leftIdx, :);
% closestToRight = centers(rightIdx, :);
% closestToTop = centers(topIdx, :);
% closestToBottom = centers(bottomIdx, :);
% 
% 
% hold on;plot(closestToLeft(1), closestToLeft(2), 'c.', 'MarkerSize', 10);
% hold on;plot(closestToRight(1), closestToRight(2), 'c.', 'MarkerSize', 10);
% hold on;plot(closestToTop(1), closestToTop(2), 'c.', 'MarkerSize', 10);
% hold on;plot(closestToBottom(1), closestToBottom(2), 'c.', 'MarkerSize', 10);


% 원본 이미지 좌표 (영상의 가장자리에서 가장 가까운 centers 점들)
% movingPoints = [closestToLeft; closestToTop; closestToRight; closestToBottom];
% bottomPoints = [closestToBottom;closestToRight];
% deltaY = diff(bottomPoints(:,2));
% deltaX = diff(bottomPoints(:,1));
% angleRad = atan2(deltaY, deltaX);
% angleDeg = rad2deg(angleRad);
% angleDeg = -angleDeg;
% 3. 이미지와 centers 회전
% rotatedImg = imrotate(Ieval, -angleDeg, 'bilinear', 'crop');
% rotatedCenters = (centers - mean(centers)) * [cos(-angleRad) -sin(-angleRad); sin(-angleRad) cos(-angleRad)] + mean(centers);

% 결과 확인
% figure;
% 원 찾기
% [centers, radii] = imfindcircles(rotatedImg, radiiRange, 'ObjectPolarity','bright', 'Sensitivity',0.60);
% [centers, radii] = imfindcircles(J2, radiiRange, 'ObjectPolarity','bright', 'Sensitivity',0.60);
% 
% % 찾은 원 표시
% hold on;plot(centers(:,1), centers(:,2), 'r.', 'MarkerSize', 20);
% title('Detected Circles');