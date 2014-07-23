img = imread('test.png')>128; % Read example input image

ctc_pruning_off = ContourTreeConnectivity(img);        % Computes contour tree with no pruning
ctc_pruning_on  = ContourTreeConnectivity(img,0.1,32); % Prune with intensity threshold 0.1 and area/volume threshold 32

figure; imagesc(img); colormap gray
axis square;
axis off;
title(['CTC(no pruning):' num2str(ctc_pruning_off) ' CTC(with pruning):' num2str(ctc_pruning_on)]);