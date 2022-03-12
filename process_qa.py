import sys; sys.path.append('/work/awilf/utils/'); from alex_utils import *
from bert import get_bert_features


def labeltxt_to_dict(txt_path, seq_len=25):
    '''
    txt_path: path to text file of form in /work/awilf/MTAG/data/social/raw/labels/{k}.txt'
    seq_len: sequence length to chop q/a/i reps to when creating array

    output: {'features': array of shape (1,6,12,3,25,768), 'intervals': [0,60]}
    '''

    '''
    raw input to lists
    raw input e.g.: 'q1': 'why did they...?'\n'a1': 'because...'\n'i1': 'so that...'\n...
    lists: [ (question, answers, incorrects) ]
    '''
    x = read_txt(txt_path).strip().split('\n')
    y = []
    curr_arr = [None,[],[]]
    for elt in x:
        if elt[0]=='q':
            if curr_arr[0] is not None:
                y.append(curr_arr)
            curr_arr = [elt.split(':',1)[1],[],[]]
        elif elt[0] == 'a':
            curr_arr[1].append(elt.split(':',1)[1])
        elif elt[0] == 'i':
            curr_arr[2].append(elt.split(':',1)[1])
    y.append(curr_arr)
    assert np.all([len(elt[1])==4 and len(elt[2])==3 for elt in y]), 'Must be 4 correct, 3 incorrect for each'

    '''
    lists to binary combinations of [ (q,a,i) ] triplets which is 72 seq_len long
    '''
    z = []
    for q,a,i in y:
        z.extend(list(product([q],a,i)))

    '''
    reshape into correct sized array
    '''
    z = ar(z).reshape(-1)
    # z = ar(z).reshape(-1).reshape(6,12,3)
    # z[0][:5], x[:10] # test

    '''
    extract bert and back pad to seq_len (or chop)
    '''
    feats = []
    for elt in z:
        temp_feats = get_bert_features(elt)
        temp_feats = temp_feats[:seq_len] # chop
        temp_feats = np.pad(temp_feats, ( (0,seq_len-temp_feats.shape[0]), (0,0) ), 'constant')
        feats.append(temp_feats)
    feats = ar(feats).reshape(1,6,12,3,25,768)
    
    return {
        'features': feats,
        'intervals': ar([[0,60]]).astype('float32'),
    }

def dir_to_csd(dir, csd_path, seq_len=25):
    '''
    dir: e.g. '/work/awilf/MTAG/data/social/raw/labels'
    csd_path: where you want the resulting csd (pk file) to go for alignment
    '''
    pk = {}

    for txt_path in tqdm(glob(join(dir, '*.txt'))):
        k = txt_path.split('/')[-1].rsplit('.',1)[0]
        v = labeltxt_to_dict(txt_path)
        pk[k] = v

    save_pk(csd_path, pk)
    
new_csd_path = '/work/awilf/MTAG/data/social/csd/new_qa.pk'
dir_to_csd(f'/work/awilf/MTAG/data/social/raw/labels', new_csd_path)

