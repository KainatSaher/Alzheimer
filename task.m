clc
clear all
close all
fpath='D:\Research\Kainat Akram\Alzheimer_Kainat\trset';
data=fullfile(fpath);
tdata=imageDatastore('D:\Research\Kainat Akram\Alzheimer_Kainat\trset','includesubfolders',true,'LabelSource','foldername');
count=tdata.countEachLabel;


  %net=googlenet;
  %net=mobilenetv2;
  %net=alexnet;  
  %net=vgg19;
  %net=inceptionresnetv2;
  %net=densenet201;
  %net=inceptionv3;
  net = alexnet;
  %net=squeezenet;
  %net=darknet19;
 % net = xception;
  
  
  
  
layers=[imageInputLayer([224 224 3]); %imag size 3 mean RGB
    net(2:end-3) %second layer to 3rd last layer tk kuch nahi karna
    fullyConnectedLayer(4) % type of classification 2 , 3 OR FOUR CLASSIFICATION
    softmaxLayer% probabality, jb fully connected ko modufy kia to next ye layer same likhni hai. ye fully connected k according khud change ho jay gi
    classificationLayer() % same, fully connected ko modufy kia to next ye layer same likhni hai. ye fully connected k according khud change ho jay gi
    ]



[traindata testdata]=splitEachLabel(tdata,0.8,'randomized');
opt=trainingOptions('adam','Maxepoch',30,'InitialLearnRate',0.001,'plot','training-progress'); %sdgm ye ye trainer/method hai. yahan sgdm, 10, 20, maxepoch o.oo1 sb change ho sakta hai
% KIS OPTION K TETEH TRAIN KARNA HAI MODEL, aur solver kon sa hai (sgdm), epoc=iterations group, plot graph ko show karta
% learner  sgdm, adam, rmsprop,adadelta, AdaGrad, Nesterov
training=trainNetwork(tdata,layers,opt);%train network function hai

allclass=[];% khud name deya, koi b name dey leyn
 allscore=[]; %khud name deya, koi b name dey leyn
 for i=1:length(testdata.Labels)
     I=readimage(testdata,i);
     [class score]=classify(training,I); %classify ka function classification kr rha
     allclass=[allclass class]; 
     allscore=[allscore score];
%      subplot(100,10,i)
%      imshow(I)
%      title(string(class))
 end
 
 %confusion matrix
 result=horzcat(testdata.Labels,allclass'); %tranpose
 figure,
 plotconfusion(testdata.Labels,allclass')
 
 