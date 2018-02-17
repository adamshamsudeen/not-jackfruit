



[//]: # (Image References)

[ImageNet1]: ./git_images/ImageNet1.jpg "ImageNet part 1"
[ImageNet2]: ./git_images/ImageNet2.jpg "ImageNet part 2"
[jackfruit]: ./git_images/jackfruit.jpg "Jackfruit"

[ToRemove1]: ./invalid/4.jpg "Inavlid Image"
[ToRemove2]: ./invalid/5.jpg "Inavlid Image"
[ToRemove3]: ./invalid/6.jpg "Inavlid Image"



# NotJackFruit-Classifier
Do you watch HBO's Silicon Valley? Because I do and I was inspired by Mr. Jian-Yang to make my own not hotdog classifier. But this is different, its the Indian version, not jackfruit?

<p align="center"> 
<img src="https://github.com/adamshamsudeen/not-jackfruit/blob/master/git_images/jackfruit.jpg?raw=true">
</p>

"What would you say if I told you there is a app on the market that tell you if you have a jackfruit or not a jackfruit." - Sasi 

# Step 1: Setting up

## Usage

#### Step #1 : Install virtualenv

`pip install virtualenv`

#### Step #2 : Create a virtualenv

`virtualenv -p python vir`

#### Step #3: Activate Virtualenv

`source ./vir/bin/activate`

#### Step #4 : Clone Repo

* Clone the Repo. *(`cd` into the `dir` after extracing the `.zip`)*

#### Step #5 : Install requirements

* Install the Requirements.

`pip install -r requirements.txt`


### If you just need to use the model without training, I have added a pretrained model. Go to step 7.



# Step 2: Collecting data
The very first step in making a classifier is to collect data. Thus we need to find images of jackfruit and not-jackfruit. 



To do this I just used [ImageNet](http://www.image-net.org/) to search for my images, since ImageNet is like a database of images. To find the hotdog images I just searched for “jackfruit”, after downloading all of the images it would me about around 1000 jackfruit images.  For the not-jackfruit images I searched for “food”, “furniture”, “people” and “pets” this give me about 4024 not-jackfruit images.

Now to actually download these images I need to get the URLs, to do that I just need to click on the tab download tab click on link called URLs
![ImageNet1]

then copy the URL of the page your on which will use in a script that we'll write to download all of these images.
![ImageNet2]

Next we need to write our scripts to download and save all of these images, here is my [python code](https://github.com/adamshamsudeen/not-jackfruit/blob/master/images/get_images.py) for saving the images. 
The function store_raw_images takes in a list of folders names where you want to save the images to from each of the links.
```
def store_raw_images(folders, links):
    pic_num = 1
    for link, folder in zip(links, folders):
        if not os.path.exists(folder):
            os.makedirs(folder)
        image_urls = str(urllib.request.urlopen(link).read())
        
        
        for i in image_urls.split('\\n'):
            try:                
                urllib.request.urlretrieve(i, folder+"/"+str(pic_num)+".jpg")
                img = cv2.imread(folder+"/"+str(pic_num)+".jpg")                         
                
                # Do preprocessing if you want
                if img is not None:
                    // do more stuff here if you want
                    cv2.imwrite(folder+"/"+str(pic_num)+".jpg",img)
                    pic_num += 1

            except Exception as e:
                    print(str(e))  

```

Next I have my main method to drive the code 

```
def main():

    links = [ 
    
        'http://image-net.org/api/text/imagenet.synset.geturls?wnid=n01318894', \
        'http://image-net.org/api/text/imagenet.synset.geturls?wnid=n03405725', \
        'http://image-net.org/api/text/imagenet.synset.geturls?wnid=n07942152', \
        'http://image-net.org/api/text/imagenet.synset.geturls?wnid=n00021265', \
        'http://image-net.org/api/text/imagenet.synset.geturls?wnid=n07697537', \
        'http://image-net.org/api/text/imagenet.synset.geturls?wnid=n12400720'
       ]

       paths = ['pets', 'furniture', 'people', 'food',  'hotdog','jackfruit']

    
    
    store_raw_images(paths, links)
```


## You can also use shell script:

Save all the liks to jack.txt and the script to download it

```
awk '{print "" $0;}' jack.txt | xargs -l1 wget
```

Delete all smaller files, unavailable images and junk files
```
find . -name "*.jpg" -size -10k -delete
```



Now just wait for it to download all of those jackfruits!!!

# Step 3: Cleaning the data
At this point we have collected our data now we just need to clean it up a bit. If you take a look at the data you will probably notice that there are some garbage images that we need to remove, images that look like one of the following

![ToRemove1]
![ToRemove2]
![ToRemove3]

To do this let write some more scripts to do this work for us, first we just need to get a copy of the images that we want to remove and place them in a folder called ‘invalid’.

```
def removeInvalid(dirPaths):
    for dirPath in dirPaths:
        for img in os.listdir(dirPath):
            for invalid in os.listdir('invalid'):
                try:
                    current_image_path = str(dirPath)+'/'+str(img)
                    invalid = cv2.imread('invalid/'+str(invalid))
                    question = cv2.imread(current_image_path)
                    if invalid.shape == question.shape and not(np.bitwise_xor(invalid,question).any()):
                        os.remove(current_image_path)
                        break

                except Exception as e:
                    print(str(e))
```
 Next I made two folders called "jackfruit" and "not-jackfruit" and placed the 'food', 'furniture', 'pets', 'people' folders in the "not-jackfruit" foldder.

# Step 4: Choosing the model.
The retrain script can retrain either Inception V3 model or a MobileNet. In this exercise, we will use a MobileNet. The principal difference is that Inception V3 is optimized for accuracy, while the MobileNets are optimized to be small and efficient, at the cost of some accuracy.

Inception V3 has a first-choice accuracy of 78% on ImageNet, but is the model is 85MB, and requires many times more processing than even the largest MobileNet configuration, which achieves 70.5% accuracy, with just a 19MB download.

Pick the following configuration options:

Input image resolution: 128,160,192, or 224px. Unsurprisingly, feeding in a higher resolution image takes more processing time, but results in better classification accuracy. We recommend 224 as an initial setting.
The relative size of the model as a fraction of the largest MobileNet: 1.0, 0.75, 0.50, or 0.25. We recommend 0.5 as an initial setting. The smaller models run significantly faster, at a cost of accuracy.
With the recommended settings, it typically takes only a couple of minutes to retrain on a laptop. You will pass the settings inside Linux shell variables. Set those shell variables as follows:
```
IMAGE_SIZE=224
ARCHITECTURE="mobilenet_0.50_${IMAGE_SIZE}"
 ```   

Other models: https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet_v1.md

To start Tensorboard

```
tensorboard --logdir tf_files/training_summaries &
```

# Step 5: Train The Neural Net
The model we choose is a mobileNet
```
python -m scripts.retrain \
  --bottleneck_dir=tf_files/bottlenecks \
  --how_many_training_steps=500 \
  --model_dir=tf_files/models/ \
  --summaries_dir=tf_files/training_summaries/"${ARCHITECTURE}" \
  --output_graph=tf_files/retrained_graph.pb \
  --output_labels=tf_files/retrained_labels.txt \
  --architecture="${ARCHITECTURE}" \
  --image_dir=images/
```


The first retraining command iterates only 500 times. You can very likely get improved results (i.e. higher accuracy) by training for longer. To get this improvement, remove the parameter --how_many_training_steps to use the default 4,000 iterations.

# Step 6: Using the retrained model

The retraining script writes data to the following two files:

tf_files/retrained_graph.pb, which contains a version of the selected network with a final layer retrained on your categories.
tf_files/retrained_labels.txt, which is a text file containing labels

To train the network we split our data into a tranining set and a test set

# Step 7: The Results
I have added a pretrined model, just use the below code without training. Dont forget to set the architecture from step 4.

```
python -m scripts.label_image \
    --graph=tf_files/retrained_graph.pb  \
    --image=images/test.jpg

```

<p align="center"> 
<img src="https://github.com/adamshamsudeen/not-jackfruit/blob/master/git_images/output.png?raw=true">
</p>
We find that we get very good results.  This would make Jian-Yang proud.

# Step 8: LICENSE
MIT


