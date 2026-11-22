from pathlib import Path
import streamlit as st
from PIL import Image
from src.core import load_model,predict
st.set_page_config(page_title='Industrial Surface Defect Inspection',page_icon='🔍',layout='centered')
st.title('🔍 Industrial Surface Defect Inspection')
st.write('Upload a product surface image to classify it as **Defective** or **Non-Defective** using a CNN.')
model_path=Path('models/defect_cnn.pt')
if not model_path.exists():
    st.error('No trained model found. Run: python scripts/make_demo_data.py && python train.py --data data/demo')
    st.stop()
@st.cache_resource
def get_model(path): return load_model(path)
model,checkpoint=get_model(str(model_path))
if checkpoint.get('demo_data'): st.warning('DEMO MODEL: trained on synthetic textures only. Predictions are illustrative, not industrially validated.')
threshold=st.slider('Defect detection threshold',0.1,0.9,0.5,0.05)
file=st.file_uploader('Upload surface image',type=['jpg','jpeg','png','bmp'])
if file:
    try:
        image=Image.open(file).convert('RGB');st.image(image,caption='Input image',use_container_width=True)
        if st.button('Inspect Product',type='primary'):
            result=predict(model,image,threshold=threshold)
            if result['label']=='defective':st.error('⚠️ DEFECTIVE PRODUCT')
            else:st.success('✅ NON-DEFECTIVE PRODUCT')
            st.metric('Model confidence',f"{result['confidence']*100:.1f}%")
            st.metric('Defect probability',f"{result['defect_probability']*100:.1f}%")
            st.progress(result['defect_probability'])
    except Exception as exc: st.error(f'Unable to read image: {exc}')
st.caption('Educational prototype | Image-level classification, not defect localization or segmentation.')
