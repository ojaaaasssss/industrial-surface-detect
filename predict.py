import argparse
from PIL import Image
from src.core import load_model,predict

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--image',required=True);ap.add_argument('--model',default='models/defect_cnn.pt');ap.add_argument('--threshold',type=float,default=0.5);args=ap.parse_args()
    model,ckpt=load_model(args.model)
    with Image.open(args.image) as im: result=predict(model,im,threshold=args.threshold)
    print('Prediction:',result['label']);print('Defect probability:',result['defect_probability']);print('Confidence:',result['confidence'])
    if ckpt.get('demo_data'):print('WARNING: Trained on synthetic demonstration images; not validated for real manufacturing.')
if __name__=='__main__':main()
