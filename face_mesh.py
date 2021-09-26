#========================================================================================================================
#========================================================================================================================
#Dependencies
import streamlit as st
import mediapipe as mp
from PIL import Image
import numpy as np
import tempfile
import time
import cv2

#========================================================================================================================
#========================================================================================================================
#Home Page

mp_drawing = mp.solutions.drawing_utils
mp_face_mesh = mp.solutions.face_mesh

DEMO_IMAGE = 'demo_assets//demo_img.jpg'
DEMO_VID = 'demo_assets//demo_video.mp4'

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

#========================================================================================================================
#========================================================================================================================
#mode_selection
app_mode = st.sidebar.selectbox('Select App Mode', ['Home','Run on Image', 'Run on Video'])

#========================================================================================================================
#Home
if app_mode == 'Home':
    st.markdown("<br/>", unsafe_allow_html=True)
    st.markdown('##### Web app to detect facial landmark features in Images as well as Videos.')
    st.markdown("<hr/>", unsafe_allow_html=True)
    st.image('demo_assets//demo_image.jpg')

#========================================================================================================================
#========================================================================================================================
#Image
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
    kpi1_text = st.markdown('')

    max_faces = st.sidebar.number_input('Max Number of Faces', value=2, min_value=1)
    st.sidebar.markdown('---')
    detection_confidence = st.sidebar.slider('Min Detection Confidence', min_value=0.0, max_value=1.0, value=0.5)
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
    min_detection_confidence = detection_confidence
    ) as face_mesh:

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

#========================================================================================================================
#========================================================================================================================
#Video

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
    st.markdown("<hr/>", unsafe_allow_html=True)


    st_frame = st.empty()
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
    fps_input = int(vid.get(cv2.CAP_PROP_FPS))

    ##Recording
    codec = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')
    out = cv2.VideoWriter('result_output.mp4', codec, fps_input, (width, height))

    st.sidebar.subheader('Upload Video')
    st.sidebar.text('Demo Video')

    st.sidebar.video(tmp_file.name)

    fps = 0
    iters = 0

    drawing_spec = mp_drawing.DrawingSpec(thickness=2, circle_radius=1)

    kpi1, kpi2, kpi3 = st.columns(3)

    with kpi1:
        st.markdown('**Framerate**')
        kpi1_text = st.markdown("0")

    with kpi2:
        st.markdown('**Detected Faces**')
        kpi2_text = st.markdown("0")
    
    with kpi3:
        st.markdown('**Image Width**')
        kpi3_text = st.markdown("0")

    st.markdown("<hr/>", unsafe_allow_html=True)
    face_ctr = 0
    
    ##Detection Logic
    with mp_face_mesh.FaceMesh(
    max_num_faces = max_faces,
    min_detection_confidence = detection_confidence,
    min_tracking_confidence = tracking_confidence
    ) as face_mesh:

        prev_time = 0

        while vid.isOpened():
            iters += 1
            ret, frame = vid.read()

            if not ret:
                continue
        
            # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(frame)
            frame.flags.writeable = True
            # frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

            face_ctr = 0

            #analyse landmarks
            if results.multi_face_landmarks:
                 ##Landmark Lines Drawing
                for face_landmarks in results.multi_face_landmarks:
                    face_ctr += 1
                    mp_drawing.draw_landmarks(
                    image = frame,
                    landmark_list = face_landmarks,
                    connections = mp_face_mesh.FACEMESH_CONTOURS,
                    landmark_drawing_spec = drawing_spec,
                    connection_drawing_spec = drawing_spec
                    )
            #FPS
            curr_time = time.time()
            fps = 1 / (curr_time - prev_time)
            prev_time = curr_time

            if record:
                out.write(frame)

            ## DASHBOARD
            kpi1_text.write(f"<h2 style='text-align: center;'>{int(fps)}</h2>", unsafe_allow_html=True)
            kpi2_text.write(f"<h2 style='text-align: center;'>{face_ctr}</h2>", unsafe_allow_html=True)
            kpi3_text.write(f"<h2 style='text-align: center;'>{width}</h2>", unsafe_allow_html=True)

            frame = cv2.resize(frame, (0,0), fx = 0.8, fy = 0.8)
            frame = img_resize(image = frame, width = 640)
            st_frame.image(frame, channels='BGR', use_column_width = True)
            
    st.text("Video Processed")
    output_video = open('result_output.mp4', 'rb')
    out_bytes = output_video.read()
    st.video(out_bytes)

    vid.release()
    out.release()
