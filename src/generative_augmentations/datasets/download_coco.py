import requests as r 
import zipfile 

from tqdm import tqdm

urls = [
"http://images.cocodataset.org/zips/train2017.zip",  # 19G, 118k images
"http://images.cocodataset.org/zips/val2017.zip",  # 1G, 5k images
"http://images.cocodataset.org/zips/test2017.zip",  # 7G, 41k images (optional)
"https://github.com/ultralytics/assets/releases/download/v0.0.0/coco2017labels-segments.zip"
]

output_dir = "data"
output_extentions = ["train", "val", "test", "masks"]
zipfiles = [output_dir + "/" + ex + ".zip" for ex in output_extentions]

for (url, file) in zip(urls, zipfiles):
    with r.get(url, stream=True) as req:
        total_size = int(req.headers.get('Content-Length', 0)) 
        chunk_size = 8192*8
        req.raise_for_status()  # Raise error for bad status
        # print(total_size / chunk_size)
        # print(int(total_size / chunk_size))
        with open(file, "wb") as f:
            for chunk in tqdm(req.iter_content(chunk_size=chunk_size), desc=f"Downloading: {file}", total=int(total_size / chunk_size)):
                if chunk:  # filter out keep-alive chunks
                    f.write(chunk)

#unpack zip files : 
for (file, dir) in zip(zipfiles, output_extentions):
    with zipfile.ZipFile(file, 'r') as zip_ref: 
        zip_ref.extractall(output_dir + dir)
