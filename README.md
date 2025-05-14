<h2>Tensorflow-Image-Segmentation-BUS-UC-Malignant (2025/05/14)</h2>
Sarah T. Arai<br>
Software Laboratory antillia.com<br><br>

This is the first experiment of Image Segmentation for BUS-UC-Malignant 
 based on 
the latest <a href="https://github.com/sarah-antillia/Tensorflow-Image-Segmentation-API">Tensorflow-Image-Segmentation-API</a>, and

a pre-augmented <a href="https://drive.google.com/file/d/14mP6Km_QKO_WKe2iGXgN_9nLMS_xtSmH/view?usp=sharing">
BUS-UC-Malignant-ImageMask-Dataset.zip</a>,
which was derived by us from 
<a href="https://data.mendeley.com/datasets/3ksd7w7jkx/1">
Mendeley Data: BUS_UC</a>
<br>
<br>
<b>Data Augmentation Strategy:</b><br>
 To address the limited size of the BUS_UC,which contains 453 images and their corresponding masks in Malignant dataset, 
 we employed <a href="./generator/ImageMaskDatasetGenerator.py">an offline augmentation tool</a> to generate a 512x512 pixels pre-augmented dataset, which supports the following augmentation methods.
<br>
<li>Vertical flip</li>
<li>Horizontal flip</li>
<li>Rotation</li>
<li>Shrinks</li>
<li>Shears</li> 
<li>Deformation</li>
<li>Distortion</li>
<li>Barrel distortion</li>
<li>Pincushion distortion</li>
<br>
Please see also the following tools <br>
<li><a href="https://github.com/sarah-antillia/Image-Deformation-Tool">Image-Deformation-Tool</a></li>
<li><a href="https://github.com/sarah-antillia/Image-Distortion-Tool">Image-Distortion-Tool</a></li>
<li><a href="https://github.com/sarah-antillia/Barrel-Image-Distortion-Tool">Barrel-Image-Distortion-Tool</a></li>
<br>
<hr>
<b>Actual Image Segmentation for Images of 512x512 pixels</b><br>
As shown below, the inferred masks look similar to the ground truth masks. <br>

<table>
<tr>
<th>Input: image</th>
<th>Mask (ground_truth)</th>
<th>Prediction: inferred_mask</th>
</tr>
<tr>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/BUS-UC-Malignant/mini_test/images/5.jpg" width="320" height="auto"></td>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/BUS-UC-Malignant/mini_test/masks/5.jpg" width="320" height="auto"></td>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/BUS-UC-Malignant/mini_test_output/5.jpg" width="320" height="auto"></td>
</tr>

<tr>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/BUS-UC-Malignant/mini_test/images/113.jpg" width="320" height="auto"></td>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/BUS-UC-Malignant/mini_test/masks/113.jpg" width="320" height="auto"></td>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/BUS-UC-Malignant/mini_test_output/113.jpg" width="320" height="auto"></td>
</tr>

<tr>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/BUS-UC-Malignant/mini_test/images/346.jpg" width="320" height="auto"></td>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/BUS-UC-Malignant/mini_test/masks/346.jpg" width="320" height="auto"></td>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/BUS-UC-Malignant/mini_test_output/346.jpg" width="320" height="auto"></td>
</tr>

</table>

<hr>
<br>
In this experiment, we used the simple UNet Model 
<a href="./src/TensorflowUNet.py">TensorflowSlightlyFlexibleUNet</a> for this BUS-UC-Malignant Segmentation Model.<br>
As shown in <a href="https://github.com/sarah-antillia/Tensorflow-Image-Segmentation-API">Tensorflow-Image-Segmentation-API</a>.
you may try other Tensorflow UNet Models:<br>

<li><a href="./src/TensorflowSwinUNet.py">TensorflowSwinUNet.py</a></li>
<li><a href="./src/TensorflowMultiResUNet.py">TensorflowMultiResUNet.py</a></li>
<li><a href="./src/TensorflowAttentionUNet.py">TensorflowAttentionUNet.py</a></li>
<li><a href="./src/TensorflowEfficientUNet.py">TensorflowEfficientUNet.py</a></li>
<li><a href="./src/TensorflowUNet3Plus.py">TensorflowUNet3Plus.py</a></li>
<li><a href="./src/TensorflowDeepLabV3Plus.py">TensorflowDeepLabV3Plus.py</a></li>

<br>

<h3>1. Dataset Citation</h3>
We used the following dataset in kaggle web site<br>
<a href="https://data.mendeley.com/datasets/3ksd7w7jkx/1">
Mendeley Data: BUS_UC</a>
<br><br>

<b>Description</b><br>
The BUS_UC dataset includes 358 benign tumor images and 453 malignant tumor images. 
The resolution of Ultrasound images is 256 × 256 pixels. 
All these images were obtained from the website Ultrasound Cases (ultrasoundcases.info),
 which does not provide ground truth images. 
 Therefore, with the help of an experienced radiologist, benign and malignant tumor images are annotated 
 for segmentation and classification task.
<br>
<br>
<b>Citation </b><br>
If you use this dataset, please cite :<br>

Ahmed Iqbal, Muhammad Sharif,<br> 
"Memory-efficient transformer network with feature fusion for breast tumor segmentation and classification task
", <br>
Engineering Applications of Artificial Intelligence, 2023.<br><br>

<b>Institutions</b><br>
COMSATS Institute of Information Technology - Wah Campus
<br><br>
<b>Categories</b><br>
Breast Cancer, Image Segmentation, Ultrasound, Image Classification
<br><br>
<b>License</b><br>
<a href="https://creativecommons.org/licenses/by/4.0/deed.en">
CC BY 4.0
</a>
<br
<br>
<h3>
<a id="2">
2 BUS-UC-Malignant ImageMask Dataset
</a>
</h3>
 If you would like to train this BUS-UC-Malignant Segmentation model by yourself,
 please download the dataset from the google drive 
 <a href="https://drive.google.com/file/d/14mP6Km_QKO_WKe2iGXgN_9nLMS_xtSmH/view?usp=sharing">
 BUS-UC-Malignant-ImageMask-Dataset.zip
</a>, expand the downloaded and put it under <b>./dataset</b> folder as shown below.
<br>
<pre>
./dataset
└─BUS-UC-Malignant
    ├─test
    │   ├─images
    │   └─masks
    ├─train
    │   ├─images
    │   └─masks
    └─valid
        ├─images
        └─masks
</pre>
On the derivation of this datata, please refer to the following Python scripts:<br>
<li><a href="">ImageMaskDatasetGenerator.py</a></li>
<li><a href=",/generator/split_master.py">split_master</a></li>
<br>
<b>BUS-UC-Malignant Dataset Statistics</b><br>
<img src ="./projects/TensorflowSlightlyFlexibleUNet/BUS-UC-Malignant/BUS-UC-Malignant_Statistics.png" width="512" height="auto"><br>
<br>
As shown above, the number of images of train and valid datasets is enough to use for a training set of our segmentation model.
<br>
<br>
<b>Train_images_sample</b><br>
<img src="./projects/TensorflowSlightlyFlexibleUNet/BUS-UC-Malignant/asset/train_images_sample.png" width="1024" height="auto">
<br>
<b>Train_masks_sample</b><br>
<img src="./projects/TensorflowSlightlyFlexibleUNet/BUS-UC-Malignant/asset/train_masks_sample.png" width="1024" height="auto">
<br>

<h3>
3 Train TensorflowUNet Model
</h3>
 We have trained BUS-UC-MalignantTensorflowUNet Model by using the following
<a href="./projects/TensorflowSlightlyFlexibleUNet/BUS-UC-Malignant/train_eval_infer.config"> <b>train_eval_infer.config</b></a> file. <br>
Please move to ./projects/TensorflowSlightlyFlexibleUNet/BUS-UC-Malignant and run the following bat file.<br>
<pre>
>1.train.bat
</pre>
, which simply runs the following command.<br>
<pre>
>python ../../../src/TensorflowUNetTrainer.py ./train_eval_infer.config
</pre>
<hr>

<b>Model parameters</b><br>
Defined a small <b>base_filters</b> and large <b>base_kernels</b> for the first Conv Layer of Encoder Block of 
<a href="./src/TensorflowUNet.py">TensorflowUNet.py</a> 
and a large num_layers (including a bridge between Encoder and Decoder Blocks).
<pre>
[model]
base_filters   = 16
base_kernels   = (9,9)
num_layers     = 8
dilation       = (3,3)
</pre>

<b>Learning rate</b><br>
Defined a small learning rate.  
<pre>
[model]
learning_rate  = 0.0001
</pre>

<b>Online augmentation</b><br>
Disabled our online augmentation.  
<pre>
[model]
model         = "TensorflowUNet"
generator     = False
</pre>

<b>Loss and metrics functions</b><br>
Specified "bce_dice_loss" and "dice_coef".<br>
<pre>
[model]
loss           = "bce_dice_loss"
metrics        = ["dice_coef"]
</pre>
<b>Learning rate reducer callback</b><br>
Enabled learing_rate_reducer callback, and a small reducer_patience.
<pre> 
[train]
learning_rate_reducer = True
reducer_factor     = 0.4
reducer_patience   = 4
</pre>


<b>Early stopping callback</b><br>
Enabled early stopping callback with patience parameter.
<pre>
[train]
patience      = 10
</pre>

<b>Epoch change inference callbacks</b><br>
Enabled epoch_change_infer callback.<br>
<pre>
[train]
epoch_change_infer       = True
epoch_change_infer_dir   =  "./epoch_change_infer"
epoch_changeinfer        = False
epoch_changeinfer_dir    = "./epoch_changeinfer"
num_infer_images         = 6
</pre>

By using this callback, on every epoch_change, the inference procedure can be called
 for 6 images in <b>mini_test</b> folder. This will help you confirm how the predicted mask changes 
 at each epoch during your training process.<br> <br> 

<b>Epoch_change_inference output at start (epoch 1,2,3)</b><br>
<img src="./projects/TensorflowSlightlyFlexibleUNet/BUS-UC-Malignant/asset/epoch_change_infer_start.png" width="1024" height="auto"><br>
<br>
<br>
<b>Epoch_change_inference output at end (epoch 98,99,100)</b><br>
<img src="./projects/TensorflowSlightlyFlexibleUNet/BUS-UC-Malignant/asset/epoch_change_infer_end.png" width="1024" height="auto"><br>
<br>
<br>

In this experiment, the training process was terminated at epoch 100.<br><br>
<img src="./projects/TensorflowSlightlyFlexibleUNet/BUS-UC-Malignant/asset/train_console_output_at_epoch_100.png" width="720" height="auto"><br>
<br>

<a href="./projects/TensorflowSlightlyFlexibleUNet/BUS-UC-Malignant/eval/train_metrics.csv">train_metrics.csv</a><br>
<img src="./projects/TensorflowSlightlyFlexibleUNet/BUS-UC-Malignant/eval/train_metrics.png" width="520" height="auto"><br>

<br>
<a href="./projects/TensorflowSlightlyFlexibleUNet/BUS-UC-Malignant/eval/train_losses.csv">train_losses.csv</a><br>
<img src="./projects/TensorflowSlightlyFlexibleUNet/BUS-UC-Malignant/eval/train_losses.png" width="520" height="auto"><br>

<br>

<h3>
4 Evaluation
</h3>
Please move to a <b>./projects/TensorflowSlightlyFlexibleUNet/BUS-UC-Malignant</b> folder,<br>
and run the following bat file to evaluate TensorflowUNet model for BUS-UC-Malignant.<br>
<pre>
./2.evaluate.bat
</pre>
This bat file simply runs the following command.
<pre>
python ../../../src/TensorflowUNetEvaluator.py ./train_eval_infer_aug.config
</pre>

Evaluation console output:<br>
<img src="./projects/TensorflowSlightlyFlexibleUNet/BUS-UC-Malignant/asset/evaluate_console_output_at_epoch_100.png" width="720" height="auto">
<br><br>Image-Segmentation-BUS-UC-Malignant

<a href="./projects/TensorflowSlightlyFlexibleUNet/BUS-UC-Malignant/evaluation.csv">evaluation.csv</a><br>

The loss (bce_dice_loss) to this BUS-UC-Malignant/test was low, and dice_coef very high as shown below.
<br>
<pre>
loss,0.0372
dice_coef,0.9616</pre>
<br>

<h3>
5 Inference
</h3>
Please move to a <b>./projects/TensorflowSlightlyFlexibleUNet/BUS-UC-Malignant</b> folder<br>
,and run the following bat file to infer segmentation regions for images by the Trained-TensorflowUNet model for BUS-UC-Malignant.<br>
<pre>
./3.infer.bat
</pre>
This simply runs the following command.
<pre>
python ../../../src/TensorflowUNetInferencer.py ./train_eval_infer_aug.config
</pre>
<hr>
<b>mini_test_images</b><br>
<img src="./projects/TensorflowSlightlyFlexibleUNet/BUS-UC-Malignant/asset/mini_test_images.png" width="1024" height="auto"><br>
<b>mini_test_mask(ground_truth)</b><br>
<img src="./projects/TensorflowSlightlyFlexibleUNet/BUS-UC-Malignant/asset/mini_test_masks.png" width="1024" height="auto"><br>

<hr>
<b>Inferred test masks</b><br>
<img src="./projects/TensorflowSlightlyFlexibleUNet/BUS-UC-Malignant/asset/mini_test_output.png" width="1024" height="auto"><br>
<br>
<hr>
<b>Enlarged images and masks </b><br>

<table>
<tr>
<th>Image</th>
<th>Mask (ground_truth)</th>
<th>Inferred-mask</th>
</tr>

<tr>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/BUS-UC-Malignant/mini_test/images/12.jpg" width="320" height="auto"></td>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/BUS-UC-Malignant/mini_test/masks/12.jpg" width="320" height="auto"></td>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/BUS-UC-Malignant/mini_test_output/12.jpg" width="320" height="auto"></td>
</tr>

<tr>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/BUS-UC-Malignant/mini_test/images/113.jpg" width="320" height="auto"></td>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/BUS-UC-Malignant/mini_test/masks/113.jpg" width="320" height="auto"></td>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/BUS-UC-Malignant/mini_test_output/113.jpg" width="320" height="auto"></td>
</tr>

<tr>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/BUS-UC-Malignant/mini_test/images/293.jpg" width="320" height="auto"></td>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/BUS-UC-Malignant/mini_test/masks/293.jpg" width="320" height="auto"></td>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/BUS-UC-Malignant/mini_test_output/293.jpg" width="320" height="auto"></td>
</tr


>
<tr>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/BUS-UC-Malignant/mini_test/images/317.jpg" width="320" height="auto"></td>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/BUS-UC-Malignant/mini_test/masks/317.jpg" width="320" height="auto"></td>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/BUS-UC-Malignant/mini_test_output/317.jpg" width="320" height="auto"></td>
</tr>
<tr>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/BUS-UC-Malignant/mini_test/images/370.jpg" width="320" height="auto"></td>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/BUS-UC-Malignant/mini_test/masks/370.jpg" width="320" height="auto"></td>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/BUS-UC-Malignant/mini_test_output/370.jpg" width="320" height="auto"></td>
</tr>
<tr>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/BUS-UC-Malignant/mini_test/images/412.jpg" width="320" height="auto"></td>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/BUS-UC-Malignant/mini_test/masks/412.jpg" width="320" height="auto"></td>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/BUS-UC-Malignant/mini_test_output/412.jpg" width="320" height="auto"></td>
</tr>
</table>
<hr>
<br>

<h3>
References
</h3>
<b>1. BUS_UC - Breast Ultrasound Dataset</b><br>
<br>
<a href="https://www.kaggle.com/datasets/orvile/bus-uc-breast-ultrasound/code">
https://www.kaggle.com/datasets/orvile/bus-uc-breast-ultrasound/code
</a>
<br>
<br>
