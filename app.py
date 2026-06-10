import streamlit as st
from PIL import Image
from predict import calculate_percentage

st.set_page_config(
    page_title="Brain Tumor Detection",
    layout="wide"
)

st.title("🧠 Brain Tumor Detection System")

uploaded_file = st.file_uploader(
    "Upload MRI Image",
    type=["png","jpg","jpeg","tif"]
)

if uploaded_file is not None:

    image = Image.open(uploaded_file)

    st.image(
        image,
        caption="Uploaded MRI Scan",
        use_container_width=True
    )

    image.save("temp.png")

    percentage, mask = calculate_percentage(
        "temp.png"
    )

    if percentage > 0:
        st.error(
            f"Tumor Detected : {percentage:.2f}%"
        )
    else:
        st.success(
            "No Tumor Detected"
        )

    st.image(
        mask[0,:,:,0] * 255,
        caption="Predicted Tumor Region",
        use_container_width=True
    )