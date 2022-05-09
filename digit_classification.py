import streamlit as st
from streamlit_drawable_canvas import st_canvas
import PIL
from PIL import Image
from digit_classification_utils import get_prediction, transform_image


def main():
    col1, col2 = st.columns(2)

    with col1:
        canvas_result = st_canvas(
            fill_color="rgba(255, 165, 0, 0.3)", 
            stroke_width=12,
            stroke_color="#FFFFFF",
            background_color="#000000",
            height=200,
            width=200,
            update_streamlit=True,
            display_toolbar=True,
            key="main",
        )

    with col2:
        if canvas_result.image_data is not None:
            canvas_img = canvas_result.image_data.astype('uint8')
            im = Image.fromarray(canvas_img)
            img = im.resize((28,28))
            img_rescaling = img.resize((200, 200), resample=PIL.Image.NEAREST)
            # st.write('Input Image')
            st.image(img_rescaling)

    if st.button('Predict'):
        tensor = transform_image(im)
        pred = get_prediction(tensor)
        st.write(f'Result: {pred[0]}')
        

if __name__ == "__main__":
    # st.set_page_config(
    #     page_title="Streamlit Drawable Canvas Demo", page_icon=":pencil2:"
    # )
    st.title("MNIST Digit Classification")
    main()
