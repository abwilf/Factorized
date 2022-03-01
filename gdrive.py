from __future__ import print_function
import os.path
from googleapiclient.discovery import build
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from apiclient.http import MediaFileUpload
import gdown

def gdrive_up(credentials_path, file_list, folder_id, token_path='/work/awilf/utils/gdrive_token.json'):
    '''
    credentials_path: json containing gdrive credentials of form {"installed":{"client_id":"<something>.apps.googleusercontent.com","project_id":"decisive-engine-<something>","auth_uri":"https://accounts.google.com/o/oauth2/auth","token_uri":"https://oauth2.googleapis.com/token","auth_provider_x509_cert_url":"https://www.googleapis.com/oauth2/v1/certs","client_secret":"<client_secret>","redirect_uris":["urn:ietf:wg:oauth:2.0:oob","http://localhost"]}}
    file_list: full path of files to upload, e.g. ['/work/awilf/tonicnet.tar']
    folder_id: id of folder you've already created in google drive (awilf@andrew account, for these credentials)

    e.g. 
    gdrive_up('gdrive_credentials.json', ['hi.txt', 'yo.txt'], '1E1ub35TDJP59rlIqDBI9SLEncCEaI4aT')

    note: if token_path does not exist, you will need to authenticate. here are the instructions

    ON MAC: ssh -N -f -L localhost:8080:localhost:8080 awilf@taro
    ON MAC (CHROME): go to link provided
    '''

    # If modifying these scopes, delete the file token.json.
    SCOPES = ['https://www.googleapis.com/auth/drive.metadata.readonly', 'https://www.googleapis.com/auth/drive.file']

    creds = None
    if os.path.exists(token_path): # UNCOMMENT THIS IF DON'T WANT TO LOG IN EACH TIME
        creds = Credentials.from_authorized_user_file(token_path, SCOPES)

    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(credentials_path, SCOPES)
            creds = flow.run_local_server(port=8080)
        with open(token_path, 'w') as token:
            token.write(creds.to_json())

    service = build('drive', 'v3', credentials=creds)

    for name in file_list:
        file_metadata = {
            'name': name,
            'parents': [folder_id]
        }
        media = MediaFileUpload(file_metadata['name'], resumable=True)
        file = service.files().create(body=file_metadata, media_body=media, fields='id').execute()


def gdrive_down(url, out_path=None):
    '''
    first, make sure file in url is available for anyone to view
    
    url should be of one of these forms:
        https://drive.google.com/file/d/195C6CoqMBYzteJIx-FFOsNFATvu5cr_z/view?usp=sharing
        https://drive.google.com/uc?id=1eGj8DSau66NiklH30UIGab55cUWR_qw9

    out_path can be None, in which case the result will be the file name from google drive saved in ./.  else, save to out_path
    '''

    if 'uc?' not in url:
        id = url.split('/')[-2]
        url = f'https://drive.google.com/uc?id={id}'
    
    gdown.download(url, out_path)

# gdrive_down('https://drive.google.com/file/d/1dAvxdsHWbtA1ZIh3Ex9DPn9Nemx9M1-L/view', out_path='/work/awilf/mfa/')

# gdrive_down('https://drive.google.com/file/d/1XEsc6rLXtjfo2rtms2GR0hDqfTiat5Zo/view?usp=sharing')

# gdrive_up('/work/awilf/utils/gdrive_credentials.json', ['pyg.sif'], '1zBldu3ipR6LtrJBxxNlaKBPW_kio6nli')
gdrive_down('https://drive.google.com/file/d/1eEdRQVgBCcq8DyasduZpMzTlCIjrekLM/view?usp=sharing')


