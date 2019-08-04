# bazingaa_unacademy_bootstrap
Bazingaa Team , Problem Statement :Unacademy
# bazingaa_unacademy_bootstrap
Bazingaa Team , Problem Statement :Unacademy

Unacademy -Identify and Remove Background from a video stream
Team Name: Bazingaa

•Steps:1.Convert video stream to frames.

2.Apply SemanticSegmentation: compute a pixel-wise mask for every object in the image. Modelsbeing considered:
a.MaskR-CNN
b.UNET

3.Filter out all other objects except ‘person’in the frame.

4.Add the frame to a video writer.

•MaskR-CNN (Reference:https://github.com/matterport/Mask_RCNN)

-Mask R-CNN process:
-After semanticsegmentation(test frame1):
-After background removal(test frame1):

•UNET:The U-Net architecture is built upon the Fully Convolutional Network and modified in a way that it yields better segmentation. Reference: https://github.com/tensorflow/tfjs-models/tree/master/mobilenet

After background removal using MobileNet API:
•User Interface: A simpleweb pageis created usingJavaScript, HTML.
•Highlights:
o Pretrained frozen TensorFlow model is being used & trained by Tensor Core through portrait datasets from Flickr.
o The TensorFlow JS Layer Model with quantization level is being used to make it lightweight(approximately 2 MB).
o Front-end being developed allows live manipulation of the video from Integrated Webcam, IP Camera, orStreaming Videos.

### Sample Result
Result 1: 
![alt text](https://github.com/Vibi007/bazingaa_unacademy_bootstrap/blob/master/images/image1.png "Result 1")

Result 2: 
![alt text](https://github.com/Vibi007/bazingaa_unacademy_bootstrap/blob/master/images/image2.png "Result 2")

Result 3: 
![alt text](https://github.com/Vibi007/bazingaa_unacademy_bootstrap/blob/master/images/image3.png "Result 3")
