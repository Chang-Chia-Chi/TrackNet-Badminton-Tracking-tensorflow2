# TrackNet-Badminton-Tracking-tensorflow2
PS: It's **not** an official implementation !
![image](https://github.com/Chang-Chia-Chi/TrackNet-Badminton-Tracking-tensorflow2/blob/main/pics/2_15_08_predict.gif)

## TrackNet
**TrackNet** is a deep learning network for higi-speed and tiny objects tracking invented by National Chiao-Tung University in Taiwan. It's a FCN model adpotes **VGG16** to generate feature map and **DeconvNet** to decode using pixel-wise classification. TrackNet could take multiple consecutive frames as input, model will learn not only object tracking but also trajectory to enhance its capability of positioning and recognition. TrackNet will generate gaussian heat map centered on ball to indicate position of the ball. Binary cross-entropy is used as loss function to compute difference between heat map of prediction and ground truth.

## Modification
### 1. Combine **ResNet** and **U-Net** to form network architecture.   
![image](https://github.com/Chang-Chia-Chi/TrackNet-Badminton-Tracking-tensorflow2/blob/main/pics/model_structure.jpg)    

|Layer|Filter size|Depth|Padding|Stride|Activation|   
|-----|-----------|-----|-------|------|----------|   
|conv1|3 x 3      |64   |2      |1     |BN+Relu   |   
|conv2|3 x 3      |64   |2      |1     |BN+Relu   |   
|resD_1|-          |32   |-      |-     |BN+Relu   |   
|resE_1 x 2|-          |32   |-      |-     |BN+Relu   |   
|resD_2|-          |64   |-      |-     |BN+Relu   | 
|resE_2 x 2|-          |64   |-      |-     |BN+Relu   |  
|resD_3|-          |128   |-      |-     |BN+Relu   |   
|resE_3 x 3|-          |128   |-      |-     |BN+Relu   |   
|resD_4|-          |256  |-      |-     |BN+Relu   |  
|resE_4 x 2|-          |256  |-      |-     |BN+Relu   |  
|resU_1 + concat|-          |128+128  |-      |-     |BN+Relu   |  
|resDE_5 x 3|-          |128   |-      |-     |BN+Relu   | 
|resU_2 + concat|-          |64+64  |-      |-     |BN+Relu   |   
|resDE_6 x 2|-          |64   |-      |-     |BN+Relu   | 
|resU_3 + concat|-          |32+32  |-      |-     |BN+Relu   |   
|resDE_7 x 2|-          |32   |-      |-     |BN+Relu   | 
|resU_4|-          |16  |-      |-     |BN+Relu   |     
|conv3|3 x 3      |64   |2      |1     |BN+Relu   |   
|conv4|3 x 3      |64   |2      |1     |BN+Relu   |   
|conv5|3 x 3      |256   |2      |1     |BN+Relu+Softmax|   

Sturcture of res-block-encoder(resE)     
|Layer|Filter size|Depth|Padding|Stride|Activation|    
|-----|-----------|-----|-------|------|----------|   
|conv1|1 x 1      |n    |0      |1     |BN+Relu   |   
|conv2|3 x 3      |n    |2      |1     |BN+Relu   |   
|conv3|1 x 1      |2n   |0      |1     |BN+Relu   |   

Sturcture of res-block-downsamping(resD)   
|Layer|Filter size|Depth|Padding|Stride|Activation|    
|-----|-----------|-----|-------|------|----------|   
|conv1|1 x 1      |n    |0      |1     |BN+Relu   |   
|conv2|3 x 3      |n    |2      |2     |BN+Relu   |   
|conv3|1 x 1      |2n   |0      |1     |BN+Relu   |   

Sturcture of res-block-decoder(resDE)     
|Layer|Filter size|Depth|Padding|Stride|Activation|    
|-----|-----------|-----|-------|------|----------|   
|conv1|1 x 1      |n    |0      |1     |BN+Relu   |   
|conv2|3 x 3      |n    |2      |1     |BN+Relu   |   
|conv3|1 x 1      |n    |0      |1     |BN+Relu   |   

Sturcture of res-block-upsamping(resU)    
|Layer|Filter size|Depth|Padding|Stride|Activation|    
|-----|-----------|-----|-------|------|----------|   
|conv1|1 x 1      |n    |0      |1     |BN+Relu   |   
|convT1|3 x 3     |n    |0      |2     |BN+Relu   |   
|conv2|1 x 1      |n    |0      |1     |BN+Relu   |   

### 2. Use Conv2Dtranspose instead for upsampling in decoder, matching structure of ResNet in encoder.
### 3. Use focal loss to help model focusing more on small ground truth.    
![image](https://github.com/Chang-Chia-Chi/TrackNet-Badminton-Tracking-tensorflow2/blob/main/pics/focal_loss.jpg)
### 4. Use consecutive 3 frames in gray scale as input image to reduce memory usage and increase training speed.
## Parameter of training
|Parameter|Value|   
|---------|-----|   
|Image size| 512 x 288|
|Heat map ball radius| 2.5 pixel|
|Batch size|2|
|Learning rate|1.0|
|Epochs|50|
|Optimizer|Adadelta|
|Number of training images| ~20k|   

## Accuracy, Precision and Recall for test.mp4
TP, FP1, FP2, TN, FN are defined as below:    
- TP: True positive, center distance of ball between prediction and ground truch is smaller than 5 pixel   
- FP1: False positive, center distance of ball between prediction and ground truch is larger than 5 pixel
- FP2: Fasle positive, if ball is not in ground truth but in prediction.
- TN: True negative.   
- FN: False positive.   
    
|Metric|Formula|Value   
|------|-------|-----   
|Accuracy|(TP+TN)/(TP+TN+FP1+FP2+FN)|0.909   
|Precision|TP/(TP+FP1+FP2)|0.939   
|Recall   |TP/(TP+FN)|0.953

## Setup
1. Clone the repository:`https://github.com/Chang-Chia-Chi/TrackNet-Badminton-Tracking-tensorflow2.git`    
2. Run `pip3 install -r requirements.txt` to install packages required.  
3. Because the model is created with `channel first`, it could be trained and tested with GPU only.
## Label
![image](https://github.com/Chang-Chia-Chi/TrackNet-Badminton-Tracking-tensorflow2/blob/main/pics/image_label.jpg)    
1. Run `python imgLabel.py` to open the program.
2. Mouse and button events are described below:   

|Mouse Event|Function|    
|-----------|--------|    
|Left click |Label center of the ball|    
|Middel click|Chancel label of the ball|    
    
|Keyboard Event|Function|   
|--------------|--------|   
|e|exit program|
|s|save csv|
|n|go to next frame|
|p|back to previous frame|
|f|go to first frame|
|l|go to last frame|
|>|fast forward 36 frames|
|<|fast backward 36 frames|   

3. If you want to load pre-labeled csv file, change `load_csv` in `imgLabel.py` to **True**.    
4. After label all frames, press `s` to save file and then press `e` to leave the program.
## Train
folder architecture:    
```
TrackNet-Badminton-Tracking-tensorflow2
|   
|___raw_data    
|       |    
|       |___ <training_videos>, <training_csvs>   
|   
|___train_data(generated from raw_data)    
|       |      
|       |___ <training_images>, <training_heat-maps>    
|   
|_____test    
|       |   
|       |___<test_video>, <test_csv>    
|   
|___weights   
        |   
        |___ <trained_weights>
```   
1. Put all videos and corresponding csvs in `raw_data` folder. Make sure name of video and csv pair is identical.   
2. Run `python video2img.py` to convert videos to trainging images and heat-maps.  
3. Run `python train.py --<args>` to train the model.
## Test
1. Put video and csv file in `test` folder and change `--video_path` and `--label_path` in `parser.py` (or indicate when type script).
2. If you don't want to compute performance of model, set `--label_path` as empty string `""`.
3. Run `python predict.py --<args>` to test the model.
## Reference
1. https://arxiv.org/abs/1907.03698   
TrackNet: A Deep Learning Network for Tracking High-speed and Tiny Objects in Sports Applications       
2. https://arxiv.org/abs/1708.02002   
Focal Loss for Dense Object Detection       
3. https://nol.cs.nctu.edu.tw:234/open-source/TrackNetv2        
TrackNetV2: Efficient TrackNet (GitLab)
