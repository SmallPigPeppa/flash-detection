PATH_TO_CKPT=byol.ckpt
python3 convert_model_to_detectron2.py --pretrained_feature_extractor ${PATH_TO_CKPT} --output_detectron_model ./detectron_model.pkl
