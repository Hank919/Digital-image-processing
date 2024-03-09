
% 計算60張水體的平均值和標準差
hueStats = [0.444886771378037	0.215663827215524];
satStats = [0.266091384780323	0.203198351537753];
valStats = [0.509337295950523	0.196241627985478];

%定義水體的HSV範圍
hueRange = [mean(allHueValues) - 1.5*std(allHueValues), mean(allHueValues) + 1.5*std(allHueValues)];
satRange = [mean(allSatValues) - 1.6*std(allSatValues), mean(allSatValues) + 1.6*std(allSatValues)];
valRange = [mean(allValValues) - 1.6*std(allValValues), mean(allValValues) + 1.6*std(allValValues)];


% 讀取待處理圖像
newImg = imread(['13.jpg']);


% 轉換到HSV空間
newImgHSV = rgb2hsv(newImg);

% 將前面提到的範圍轉為mask，之後套用在圖片上
hueMask = (newImgHSV(:,:,1) >= hueRange(1)) & (newImgHSV(:,:,1) <= hueRange(2));
satMask = (newImgHSV(:,:,2) >= satRange(1)) & (newImgHSV(:,:,2) <= satRange(2));
valMask = (newImgHSV(:,:,3) >= valRange(1)) & (newImgHSV(:,:,3) <= valRange(2));

% 將3個mask合起來
waterMask = hueMask & satMask & valMask;

% opening closing操作
se = strel('disk', 5);
waterMaskOpened = imopen(waterMask, se); % Opening 操作
waterMaskClosed = imclose(waterMaskOpened, se); % Closing 操作
waterMaskFinal = waterMaskClosed;

% 使用 K-means 分割原圖
k = 25; % 區塊數

reshapedImg = double(reshape(newImg, [], 3)); 
[clusterIdx, ~] = kmeans(reshapedImg, k, 'Distance', 'sqEuclidean', 'MaxIter', 1000, 'Replicates', 3);
clusteredImg = reshape(clusterIdx, size(newImg, 1), size(newImg, 2));

clusteredImgColor = zeros(size(newImg, 1), size(newImg, 2), 3, 'uint8');

% 為每個類別分配顏色
colors = lines(k); 
for j = 1:k
    clusteredImgColor(:,:,1) = clusteredImgColor(:,:,1) + uint8(clusteredImg == j)*colors(j, 1)*255;
    clusteredImgColor(:,:,2) = clusteredImgColor(:,:,2) + uint8(clusteredImg == j)*colors(j, 2)*255;
    clusteredImgColor(:,:,3) = clusteredImgColor(:,:,3) + uint8(clusteredImg == j)*colors(j, 3)*255;
end

% 初始化水體區塊標記陣列
waterBodyFlag = false(k, 1);

% 遍歷每個區塊，判別是否為水體
for j = 1:k
   
    currentClusterMask = clusteredImg == j;
    
    % 計算當前區塊水體的覆蓋率
    overlapArea = sum(sum(currentClusterMask & waterMask));
    totalArea = sum(sum(currentClusterMask));
    coverageRatio = overlapArea / totalArea;
    
    % 如果覆蓋率超過70%，則標記為水體區塊
    if coverageRatio > 0.7
        waterBodyFlag(j) = true;
    end
end


waterBodyMarked = zeros(size(newImg, 1), size(newImg, 2));

% 根據waterBodyFlag標記水體區域為1
for j = 1:k
    if waterBodyFlag(j)
        waterBodyMarked(clusteredImg == j) = 1;
    end
end

% opening closing處理
se = strel('disk',1);
waterMaskOpened1 = imopen(waterBodyMarked, se); % Opening 操作
waterMaskClosed1 = imclose(waterMaskOpened1, se); % Closing 操作

% 最终结果
waterMaskFinal1 = waterMaskClosed1;

M = 100;
filename = sprintf('C:/Users/User/Desktop/DIP/final project/training_dataset/output/%d.jpg', M); % 改用正斜線
imwrite(waterMaskFinal1, filename); % 保存圖片
% 顯示結果
figure;
subplot(2, 2, 1);
imshow(newImg);
title('Original Image');

subplot(2, 2, 2);
imshow(waterMask);
title('Only used HSV mask');

subplot(2, 2, 3);
imshow(waterBodyMarked);
title('Final mask without opening and closing');

subplot(2, 2, 4);
imshow(waterMaskFinal1);
title('Final mask with opening and closing');

