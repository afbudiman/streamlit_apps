import streamlit as st
from streamlit_drawable_canvas import st_canvas
import cv2
from digit_classification_utils import get_prediction, transform_image


def main():
    canvas_result = st_canvas(
        fill_color="rgba(255, 165, 0, 0.3)", 
        stroke_width=12,
        stroke_color="#FFFFFF",
        background_color="#000000",
        height=200,
        width=200,
        key="main",
    )

    if canvas_result.image_data is not None:
        img = cv2.resize(canvas_result.image_data.astype('uint8'), (28, 28))
        img_rescaling = cv2.resize(img, (200, 200), interpolation=cv2.INTER_NEAREST)
        st.write('Input Image')
        st.image(img_rescaling)

    if st.button('Predict'):
        test_x = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        tensor = transform_image(test_x)
        pred = get_prediction(tensor)
        st.write(f'Result: {pred[0]}')
        

if __name__ == "__main__":
    st.set_page_config(
        page_title="Streamlit Drawable Canvas Demo", page_icon=":pencil2:"
    )
    st.title("Drawable Canvas Demo")
    main()
