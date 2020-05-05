# LIV360SV 

We present a workflow for extracting and classifying advertisements located within street level images. We use a seamless scene segmentation network to identify regions within street-level images where advertisements are located. To subsequently classify the extracted advertisements we train a MobileNet-V2 to differentiate advertisement types using data scraped from Google Images. We introduce the Liverpool 360 Street View (LIV360SV) dataset for evaluating our workflow. The dataset contains 26,645, 360 degree, street-level images collected via cycling with a GoPro Fusion 360 camera.

## Workflow

![Architecture](./img/architecture.png)





WorkFlow components:

1.) Process images using seamseg
2.) Extract individual ads and areas using pre-processing scritp
3.) Get STN parameters for each ad
4.) Transform each ad into
5.) Classify
 
