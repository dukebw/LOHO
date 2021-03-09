# LOHO: Latent Optimization of Hairstyles via Orthogonalization [CVPR'21]

![Hairstyle transfer samples synthesized by LOHO.](imgs/teaser.jpeg "LOHO Teaser")

This directory contains the code for running LOHO framework.

The sub-folders are:
* **networks**: Contains scripts for Graphonomy, VGG16 and StyleGANv2
* **losses**: Contains scripts for computing different losses
* **datasets**: Contains scripts for preparing the images and masks required to run LOHO
* **data**:
	* images: Contains FFHQ images at 1024px
	* masks: Contains masks at 128px, extracted from Graphonomy, corresponding to the images
	* softmasks: Contains pickle files used to perform "soft-blending" as a post-processing step at 512px
	* backgrounds: Contains images at 256px with foreground inpainted 
	* results: Folders that store output files
* **checkpoints**: Folder to store checkpoints

In order to run LOHO, you have to download the necessary model checkpoints. We provide instructions to download checkpoints:
* Download the checkpoints for Graphonomy and StyleGANv2 from: https://drive.google.com/drive/folders/10goJlS18m9si3fBOmQKjszS2m9tfjfp7?usp=sharing
* Place the files in the folder: checkpoints/ 

Next, running LOHO requires relevant python and CUDA packages. Please run requirements.sh to install necessary packages via conda. Alternatively, you can use pip to install the packages.

Finally, execute loho.py and mention the flags --image1, --image2, --image3. We provide examples below for you to try LOHO. We also provide loop.sh that goes over all combinations and stores the outputs under data/results.

You can use the following specifications:
* **python loho.py --image1 67172.jpg --image2 02602.jpg --image3 67172.jpg**
* **python loho.py --image1 00761.jpg --image2 00761.jpg --image3 00018.jpg**
* **python loho.py --image1 52364.jpg --image2 52364.jpg --image3 19501.jpg**
* **python loho.py --image1 17754.jpg --image2 17658.jpg --image3 00148.jpg**
* **python loho.py --image1 46826.jpg --image2 08244.jpg --image3 10446.jpg**

#### To cite this paper, use the following: 

```
 @inproceedings{saha2021LOHO,
   title={LOHO: Latent Optimization of Hairstyles via Orthogonalization},
   author={Saha, Rohit and Duke, Brendan and Shkurti, Florian and Taylor, Graham, and Aarabi, Parham},
   booktitle={CVPR},
   year={2021}
 }
 ```
