# Set up virtual enviroments
## Windows
```bash
pip install virtualenv
virtualenv --python <directory> venv
.\venv\Scripts\activate
```
## Linux
```bash
apt-get install python3-venv
python3 -m venv venv
source venv/bin/activate
```
# Install Requirements
```bash
pip install -r requirements.txt
```

# File Locations
```
image-caption
│   readme.md
│   requirements.txt
│
├───captionist
│   │   data_loader.py
│   │   data_util.py
│   │   main.py
│   │   model.py
│   │   model_loader.py
│   │   model_util.py
│   │   trainer.py
│   │   utils.py
│   │   vocab_builder.py
│   │
│   └───model_data
│           saved_model.pth
│
└───data
    │   captions.txt
    │
    ├───Images
    └───test_images
```

# Run the project
## Train
```bash
python main.py --train 
```

## Test
```bash
python main.py
```

## Data Location
Flicker8k: https://www.kaggle.com/datasets/adityajn105/flickr8k <br />
Saved model: https://drive.google.com/file/d/1K2sstgLP9Fdws9G2MVQ0XU_3o1G3XMgy/view?usp=sharing