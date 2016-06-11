clear;close all;
%% settings
folder = 'Test/Set5';
savepath = 'test.h5';
size_input = 33;
size_label = 21;
scale = 3;
stride = 21 ;

%% initialization
data = zeros(1113, 1, size_input, size_input);
label = zeros(1113, 1, size_label, size_label);
padding = abs(size_input - size_label)/2;
count = 0;

%% generate data
filepaths = dir(fullfile(folder,'*.bmp'));
    
for i = 1 : length(filepaths)
    
    image = imread(fullfile(folder,filepaths(i).name));
    image = rgb2ycbcr(image);    
    image = im2double(image(:, :, 1));    
    im_label = modcrop(image, scale);
    [hei,wid] = size(im_label);
    im_input = imresize(imresize(im_label,1/scale,'bicubic'),[hei,wid],'bicubic');    
    for x = 1 : stride : hei-size_input+1
        for y = 1 :stride : wid-size_input+1
            
            subim_input = im_input(x : x+size_input-1, y : y+size_input-1);
            subim_label = im_label(x+padding : x+padding+size_label-1, y+padding : y+padding+size_label-1);

            count=count+1;
            data(count, 1, :, :) = uint8(subim_input*255);
            label(count, 1, :, :) = uint8(subim_label*255);
        end
    end
end

order = randperm(count);
x_test = uint8(data(order, 1, :, :));
y_test = uint8(label(order, 1, :, :)); 
save('test.mat', 'x_test', 'y_test');