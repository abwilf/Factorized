import sys; sys.path.append('/work/awilf/utils/'); from alex_utils import *

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

    import cv2
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

def get_densenet_image(img, model, transforms, Image, torch):
    '''
    img is vector of shape [height_pixels, width_pixels, 3]
    model is pretrained densenet161 model
    transforms, Image are passed in libraries so I can include this in general alex_utils
    
    '''
    # Open image
    input_image = Image.fromarray(img)

    # Preprocess image
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    input_tensor = preprocess(input_image)
    input_batch = input_tensor.unsqueeze(0)  # create a mini-batch as expected by the model

    # move the input and model to GPU for speed if available
    if torch.cuda.is_available():
        input_batch = input_batch.to("cuda")

    features = model.extract_features(input_batch)
    features = features.mean(-1).mean(-1).squeeze().detach().cpu()
    del input_batch

    torch.cuda.empty_cache()

    return features

def get_densenet_video(vid):
    global model
    import torch
    import torchvision.transforms as transforms
    from PIL import Image
    from densenet_pytorch import DenseNet 
    if model is None:
        model = DenseNet.from_pretrained("densenet161")
        if torch.cuda.is_available():
            model.to("cuda")
        model.eval()

    all_feats = []
    for img in vid:
        features = get_densenet_image(img, model, transforms, Image, torch)
        all_feats.append(features)

    all_feats = torch.vstack(all_feats)
    
    return all_feats

def get_densenet_videopath(vidpath, desired_fps=1):
    # in: path to video mp4 file, fps you want
    # out: 
    #   densenet features extracted over those frames: [num_frames=fps*num_secs, 2208]
    #   intervals

    vid, intervals = video_to_frames(vidpath,desired_fps)
    if vid is None: # video was not able to be processed
        return None,None
    features = get_densenet_video(vid)
    return features, intervals

def get_densenet_features(video_dir, desired_fps=1, temp_save_path='temp.pk'):
    pk = load_pk(temp_save_path)
    if pk is None:
        pk = {}
    
    paths = glob(join(video_dir, '*.mp4'))
    keys = [elt.split('/')[-1].split('.mp4')[0] for elt in paths]

    global model, DenseNet, torch
    from densenet_pytorch import DenseNet 
    import torch
    model = DenseNet.from_pretrained("densenet161")
    if torch.cuda.is_available():
        model.to("cuda")
    model.eval()

    count = 0
    for path,key in tqdm(lzip(paths,keys)):
        if key in pk:
            continue
        
        features, intervals = get_densenet_videopath(path, desired_fps)
        if features is not None:
            pk[key] = {
                'features': features,
                'intervals': intervals
            }
        count += 1
        # if count > 2:
        #     break
        if count % 15 == 0: # save every so often
            save_pk(temp_save_path, pk)
    
    save_pk(temp_save_path, pk)
    return pk


if __name__ == '__main__':
    desired_fps = 1
    # path = '/work/awilf/covarep/mp4s/00m9ssEAnU4.mp4'
    video_dir = '/work/awilf/social_iq_raw/vision/raw/'
    video_dir = 'data/mosi/raw/Raw/Video/Full/'
    pk = get_densenet_features(video_dir)
    a = 2

