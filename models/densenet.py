import sys; sys.path.append('/work/awilf/utils/'); from alex_utils import *
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image 

from transformers import BeitFeatureExtractor, BeitModel
import transformers

import torch
import torchvision
import cv2

np.cat = np.concatenate

def video_to_frames(path,desired_fps=1):
    '''
    in: path to video
    out: 
        vid: np array of shape # frames, height_pixels, width_pixels, 3
            intervals
            frames_per_sec
            duration
    '''
    vidcap = cv2.VideoCapture(path)
    
    images = []
    
    success,image = vidcap.read()
    images.append(np.expand_dims(image,0))

    while success:
        success,image = vidcap.read()
        if success:
            images.append(np.expand_dims(image,0))

    images = np.cat(images)
    fps = vidcap.get(cv2.CAP_PROP_FPS)
    if fps == 0:
        return None,None
    assert desired_fps <= fps, f'Desired frames per second {desired_fps} must be less than or equal to fps of video, {fps}'

    frame_count = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = frame_count/fps
    
    starts = np.expand_dims(np.linspace(0,duration,images.shape[0])[:-1],0)
    ends = np.expand_dims(np.linspace(0,duration,images.shape[0])[1:],0)
    intervals = np.cat([starts,ends]).T
    
    # trim down to desired fps by taking the first frame in each portion
    secs_per_frame = 1/desired_fps
    new_intervals = []
    new_intervals = np.vstack([np.arange(0,duration,secs_per_frame)[:-1], np.arange(0,duration,secs_per_frame)[1:]]).T
    image_idxs = np.linspace(0,intervals.shape[0], new_intervals.shape[0]).astype(np.int32)
    new_images = images[image_idxs]

    return new_images, new_intervals

# def get_image_batch(img, model):
#     '''
#     img is vector of shape [height_pixels, width_pixels, 3]
#     model is pretrained densenet161 model
#     transforms, Image are passed in libraries so I can include this in general alex_utils
    
#     '''
#     # Open image
#     input_image = Image.fromarray(img)

#     # Preprocess image
#     preprocess = transforms.Compose([
#         transforms.Resize(256),
#         transforms.CenterCrop(224),
#         transforms.ToTensor(),
#         transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
#     ])
#     input_tensor = preprocess(input_image)
#     input_batch = input_tensor.unsqueeze(0)  # create a mini-batch as expected by the model
#     return input_batch

bs = None
def get_densenet_videopath(vidpath, model, feature_extractor, desired_fps=1):
    # in: path to video mp4 file, fps you want
    # out: 
    #   densenet features extracted over those frames: [num_frames=fps*num_secs, 2208]
    #   intervals

    vid, intervals = video_to_frames(vidpath,desired_fps)
    if vid is None: # video was not able to be processed
        return None,None

    # orig_batch = torch.vstack([get_image_batch(img, model) for img in vid])
    orig_batch = torch.Tensor(vid)

    global bs
    bs = 1
    if bs is None: # initially, determine what batch size (number of frames) will be acceptable to GPU memory
        bs = 1
        batch = [orig_batch]
    
        while True:
            try:
                all_feats = []
                for sb in batch:
                    with torch.no_grad():
                        inputs = feature_extractor(images=list(sb), return_tensors="pt")
                        # inputs = {k: v.to('cuda') for k,v in inputs.items()}
                        features = model(**inputs).pooler_output

                    all_feats.append(features)
                    del sb
                    del inputs
                    # torch.cuda.empty_cache()
                break
            except:
                del batch
                batch = torch.split(orig_batch, int(np.ceil(orig_batch.shape[0]/bs)))
                # torch.cuda.empty_cache()

            bs += 1
        
        bs += 3 # to be safe for some batches containing more info than others
    
    batch = torch.split(orig_batch, int(np.ceil(orig_batch.shape[0]/bs)))
    batch = [orig_batch]
    all_feats = []
    for sb in batch:
        with torch.no_grad():
            inputs = feature_extractor(images=list(sb), return_tensors="pt")
            # inputs = {k: v.to('cuda') for k,v in inputs.items()}
            features = model(**inputs).pooler_output

        all_feats.append(features)
        del sb
        del inputs
        # torch.cuda.empty_cache()
    
    features = torch.vstack(all_feats).detach().cpu().numpy()
    return features, intervals


def get_densenet_features(video_dir, desired_fps=1, temp_save_path='temp.pk'):
    # pk = load_pk(temp_save_path)
    pk = None
    if pk is None:
        pk = {}
    
    paths = glob(join(video_dir, '*.mp4'))
    keys = [elt.split('/')[-1].split('.mp4')[0] for elt in paths]

    feature_extractor = BeitFeatureExtractor.from_pretrained("microsoft/beit-base-patch16-224-pt22k-ft22k")
    model = BeitModel.from_pretrained("microsoft/beit-base-patch16-224-pt22k-ft22k")
    # model = model.to('cuda')
    model.eval()

    count = 0
    for path,key in tqdm(lzip(paths,keys)):        
        count += 1
        if key in pk:
            continue

        features, intervals = get_densenet_videopath(path, model, feature_extractor, desired_fps)

        if features is not None:
            pk[key] = {
                'features': features,
                'intervals': intervals
            }
        
        # if count % 50 == 0: # save every so often
        #     save_pk(temp_save_path, pk) # sometimes this line fails and corrupts memory and shuts down my computer
        #     save_pk('temp.pk', pk)
    
    save_pk(temp_save_path, pk)
    return pk


if __name__ == '__main__':
    desired_fps = 1
    # path = '/work/awilf/covarep/mp4s/00m9ssEAnU4.mp4'
    # video_dir = '/work/awilf/social_iq_raw/vision/raw/'
    # video_dir = 'data/mosi/raw/Raw/Video/Full/'
    # video_dir = '/work/awilf/MTAG/data/social/raw/video'
    video_dir = '/work/awilf/MTAG/data/bmw/mp4s/'
    pk = get_densenet_features(video_dir, 1, '/work/awilf/MTAG/beit_bmw.pk')
    a = 2

