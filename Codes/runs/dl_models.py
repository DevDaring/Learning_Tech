from huggingface_hub import snapshot_download as s
print(s('ai4bharat/indictrans2-en-indic-1B', revision='10e65a9951a1e922cd109a95e8aba9357b62144b'))
print(s('ai4bharat/indictrans2-indic-en-1B', revision='ac3daf0ecd37be3b6957764a9179ab2b07fa9d6a'))
