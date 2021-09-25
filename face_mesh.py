import streamlit as st
import mediapipe as mp
from PIL import Image
import numpy as np
import tempfile
import time
import cv2

mp_drawing = mp.solutions.drawing_utils
mp_face_mesh = mp.solutions.face_mesh

DEMO_IMAGE = 'demo_assets/demo_img.jpg'
DEMO_VID = 'demo_video.mp4'

st.title('Face Mesh Detection')

st.markdown(
    """
    <style>
    [data-testid="stSidebar"][aria-expanded="true"] > div:first-child{
        width:350px
    }
    [data-testid="stSidebar"][aria-expanded="false"] > div:first-child{
        width:350px
        margin-left: -350px
    }
    </style>
    """,
    unsafe_allow_html = True,
)
st.sidebar.title("FaceMesh Sidebar")
st.sidebar.subheader('Parameters')

@st.cache()
#Image Resizing
def img_resize(image, width=None, height=None, inter = cv2.INTER_AREA):
    dim = None
    (h,w) = image.shape[:2]

    if width is None and height is None:
        return image
    
    if width is None:
        r = width/float(w)
        dim = (int(w*r), height)

    else:
        r = width/float(w)
        dim = (width, int(h*r))
    
    #resize
    resized = cv2.resize(image, dim, interpolation=inter)

    return resized

#mode_selection
app_mode = st.sidebar.selectbox('Select App Mode', ['Home','Run on Image', 'Run on Video'])

if app_mode == 'Run on Image':
    drawing_spec = mp_drawing.DrawingSpec(thickness=2, circle_radius=1)

    st.sidebar.markdown('---')
    st.markdown(
    """
    <style>
    [data-testid="stSidebar"][aria-expanded="true"] > div:first-child{
        width:350px
    }
    [data-testid="stSidebar"][aria-expanded="false"] > div:first-child{
        width:350px
        margin-left: -350px
    }
    </style>
    """,
    unsafe_allow_html = True,
    )

    st.subheader("**Detected Faces**")
    kpi1_text = st.markdown('0')

    max_faces = st.sidebar.number_input('Max Number of Faces', value=2, min_value=1)
    st.sidebar.markdown('---')
    detection_confidence = st.sidebar.slider('Miin Detection Confidence', min_value=0.0, max_value=1.0, value=0.5)
    st.sidebar.markdown('---')

    img_file_buffer = st.sidebar.file_uploader("Upload Image (jpg / jpeg / png)", type = ["jpg", "jpeg", "png"])
    if img_file_buffer is not None:
        img = np.array(Image.open(img_file_buffer))

    else:
        demo_img = DEMO_IMAGE
        img = np.array(Image.open(DEMO_IMAGE))

    st.sidebar.text('Current Image')
    st.sidebar.image(img)

    face_ctr = 0

    ##Dashboard
    with mp_face_mesh.FaceMesh(
    static_image_mode = True,
    max_num_faces = max_faces,
    min_detection_confidence = detection_confidence) as face_mesh:

        results = face_mesh.process(img)
        out_image = img.copy()

        ##Landmark Lines Drawing
        for face_landmarks in results.multi_face_landmarks:
            face_ctr += 1

            mp_drawing.draw_landmarks(
            image = out_image,
            landmark_list = face_landmarks,
            connections = mp_face_mesh.FACEMESH_CONTOURS,
            landmark_drawing_spec = drawing_spec)

            kpi1_text.write(f"<h2 style='text-align: center;'>{face_ctr}</h2>", unsafe_allow_html=True)
        
        st.subheader("Result: ")
        st.image(out_image, use_column_width=True)

if app_mode == 'Run on Video':
    
    st.set_option('deprecation.showfileUploaderEncoding', False)

    use_webcam = st.sidebar.button('Use Webcam')
    record = st.sidebar.checkbox('Record Video')

    if record:
        st.checkbox('Recording', value=True)

    st.markdown(
    """
    <style>
    [data-testid="stSidebar"][aria-expanded="true"] > div:first-child{
        width:350px
    }
    [data-testid="stSidebar"][aria-expanded="false"] > div:first-child{
        width:350px
        margin-left: -350px
    }
    </style>
    """,
    unsafe_allow_html = True,
    )

    max_faces = st.sidebar.number_input('Max Number of Faces', value=5, min_value=1)
    st.sidebar.markdown('---')
    detection_confidence = st.sidebar.slider('Min Detection Confidence', min_value=0.0, max_value=1.0, value=0.5)
    tracking_confidence = st.sidebar.slider('Min Tracking Confidence', min_value=0.0, max_value=1.0, value=0.5)
    st.sidebar.markdown('---')

    st.markdown("## Output")

    st_frame = st.empty
    video_file_buffer = st.sidebar.file_uploader("Upload a Video", type=['mp4', 'avi', 'mov', 'wmv', 'flv', 'mkv'])
    tmp_file = tempfile.NamedTemporaryFile(delete=False)

    ## Get Video
    if not video_file_buffer:
        if use_webcam:
            vid = cv2.VideoCapture(0)
        else:
            vid = cv2.VideoCapture(DEMO_VID)
            tmp_file.name = DEMO_VID
    else:
        tmp_file.write(video_file_buffer.read())
        vid = cv2.VideoCapture(tmp_file.name)

    width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))


    face_ctr = 0

    ##Dashboard
    with mp_face_mesh.FaceMesh(
    static_image_mode = True,
    max_num_faces = max_faces,
    min_detection_confidence = detection_confidence) as face_mesh:

        results = face_mesh.process(img)
        out_image = img.copy()

        ##Landmark Lines Drawing
        for face_landmarks in results.multi_face_landmarks:
            face_ctr += 1

            mp_drawing.draw_landmarks(
            image = out_image,
            landmark_list = face_landmarks,
            connections = mp_face_mesh.FACEMESH_CONTOURS,
            landmark_drawing_spec = drawing_spec)

            kpi1_text.write(f"<h2 style='text-align: center;'>{face_ctr}</h2>", unsafe_allow_html=True)
        
        st.subheader("Result: ")
        st.image(out_image, use_column_width=True)