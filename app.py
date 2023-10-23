import boto3  # AWS SDK to manage AWS services.
import gc  # Garbage collector.
import joblib  # Load StandardScaler.
import librosa  # Audio analysis.
import numpy as np  # Data wrangling.
import pydub  # Manipulate audio.
import queue  # Queue data structure.
import random  # Random variable generators.
import s3fs  # Pythonic file interface to S3.
import streamlit as st  # Streamlit.
import time  # Time access and conversions.
import wave  # Interface to the WAV sound format.
from datetime import datetime  # Combination of a date and a time.
from io import BytesIO  # A binary stream using an in-memory bytes buffer.
from memory_profiler import (
    profile,
)  # A module for monitoring memory usage of a python program.

# To encode target labels with value between 0 and n_classes-1.
# To perform standardization by centering and scaling.
from sklearn.preprocessing import StandardScaler
from statistics import mode  # Find the most likely predicted emotion.
from tensorflow.keras.models import load_model  # To load the model.
from streamlit_webrtc import (
    RTCConfiguration,
    WebRtcMode,
    webrtc_streamer,
)  # Handling and transmitting real-time video/audio streams over the network with Streamlit.

# Session states.
if "bucket_name" not in st.session_state:
    st.session_state["bucket_name"] = st.secrets["bucket_name"]

# Create the connection to S3.
if "s3" not in st.session_state:
    st.session_state["s3"] = s3fs.S3FileSystem(anon=False)

if "client" not in st.session_state:
    st.session_state["client"] = boto3.client("s3")

if "recordings_path" not in st.session_state:
    st.session_state["recordings_path"] = st.secrets["recordings_path"]

if "model_path" not in st.session_state:
    st.session_state["model_path"] = st.secrets["model_path"]

if "standard_scaler_path" not in st.session_state:
    st.session_state["standard_scaler_path"] = st.secrets["standard_scaler_path"]

# if "X_train_path" not in st.session_state:
#     st.session_state["X_train_path"] = st.secrets["X_train_path"]

# if "X_valid_path" not in st.session_state:
#     st.session_state["X_valid_path"] = st.secrets["X_valid_path"]

# if "X_test_path" not in st.session_state:
#     st.session_state["X_test_path"] = st.secrets["X_test_path"]

if "RTC_CONFIGURATION" not in st.session_state:
    st.session_state["RTC_CONFIGURATION"] = RTCConfiguration(
        {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
    )

if "emotions_classes" not in st.session_state:
    st.session_state["emotions_classes"] = [
        "angry",
        "calm",
        "disgust",
        "fear",
        "happy",
        "neutral",
        "sad",
        "surprise",
    ]

if "pred_emotion" not in st.session_state:
    st.session_state["pred_emotion"] = ""

# Prompts used in training data.
if "prompts" not in st.session_state:
    st.session_state["prompts"] = [
        "Kids are talking by the door",
        "Dogs are sitting by the door",
        "It's eleven o'clock",
        "That is exactly what happened",
        "I'm on my way to the meeting",
        "I wonder what this is about",
        "The airplane is almost full",
        "Maybe tomorrow it will be cold",
        "I think I have a doctor's appointment",
        "Say the word apple",
    ]

# Emotions used in training data.
if "emotion_dict" not in st.session_state:
    st.session_state["emotion_dict"] = {
        "angry": "angry 😡",
        "calm": "calm 😌",
        "disgust": "disgusted 🤢",
        "fear": "scared 😨",
        "happy": "happy 😆",
        "neutral": "neutral 🙂",
        "sad": "sad 😢",
        "surprise": "surprised 😳",
    }

if "initial_styling" not in st.session_state:
    st.session_state["initial_styling"] = True

if "particle" not in st.session_state:
    st.session_state["particle"] = "👋🏻"

if "prompt" not in st.session_state:
    st.session_state["prompt"] = ""

if "emotion" not in st.session_state:
    st.session_state["emotion"] = ""

if "is_prompt" not in st.session_state:
    st.session_state["is_prompt"] = False

if "is_emotion" not in st.session_state:
    st.session_state["is_emotion"] = False

if "is_first_time_prompt" not in st.session_state:
    st.session_state["is_first_time_prompt"] = True

if "dt_string" not in st.session_state:
    st.session_state["dt_string"] = ""

if "calculated_time" not in st.session_state:
    st.session_state["calculated_time"] = False

if "recording_name" not in st.session_state:
    st.session_state["recording_name"] = ""

if "record" not in st.session_state:
    st.session_state["record"] = False

# Create unique string identifier that will be used when naming/storing the audio files.
if not (st.session_state["calculated_time"]):
    st.session_state["calculated_time"] = True
    now = datetime.now()
    dt_string = now.strftime("%Y-%m-%d_%H:%M:%S.%f_%p")
    st.session_state["dt_string"] = dt_string


# Next 4 functions are Audio Data Augmentation:
# Noise Injection.
def inject_noise(data, sampling_rate=0.035, threshold=0.075, random=False):
    if random:
        sampling_rate = np.random.random() * threshold
    noise_amplitude = sampling_rate * np.random.uniform() * np.amax(data)
    augmented_data = data + noise_amplitude * np.random.normal(size=data.shape[0])
    del sampling_rate
    del noise_amplitude
    gc.collect()
    return augmented_data


# Pitching.
def pitching(data, sampling_rate, pitch_factor=0.7, random=False):
    if random:
        pitch_factor = np.random.random() * pitch_factor
    return librosa.effects.pitch_shift(y=data, sr=sampling_rate, n_steps=pitch_factor)


# Zero crossing rate.
def zero_crossing_rate(data, frame_length, hop_length):
    return np.squeeze(
        librosa.feature.zero_crossing_rate(
            y=data, frame_length=frame_length, hop_length=hop_length
        )
    )


# Root mean square.
def root_mean_square(data, frame_length=2048, hop_length=512):
    return np.squeeze(
        librosa.feature.rms(y=data, frame_length=frame_length, hop_length=hop_length)
    )


# Mel frequency cepstral coefficients.
def mel_frequency_cepstral_coefficients(
    data, sampling_rate, frame_length=2048, hop_length=512, flatten: bool = True
):
    return (
        np.squeeze(librosa.feature.mfcc(y=data, sr=sampling_rate).T)
        if not flatten
        else np.ravel(librosa.feature.mfcc(y=data, sr=sampling_rate).T)
    )


# Combined audio data feature extraction.
def feature_extraction(data, sampling_rate, frame_length=2048, hop_length=512):
    result = np.array([])
    result = np.hstack(
        (
            result,
            zero_crossing_rate(data, frame_length, hop_length),
            root_mean_square(data, frame_length, hop_length),
            mel_frequency_cepstral_coefficients(
                data, sampling_rate, frame_length, hop_length
            ),
        )
    )
    return result


# Duration and offset act as placeholders because there is no audio in start and the ending of
# each audio file is normally below three seconds.
# Combine audio data augmentation and audio data feature extraction.
@profile
def get_features(file_path, duration=2.5, offset=0.6):
    data, sampling_rate = librosa.load(file_path, duration=duration, offset=offset)

    # No audio data augmentation.
    audio_1 = feature_extraction(data, sampling_rate)
    audio = np.array(audio_1)
    del audio_1

    # Inject Noise.
    noise_audio = inject_noise(data, random=True)
    audio_2 = feature_extraction(noise_audio, sampling_rate)
    audio = np.vstack((audio, audio_2))
    del noise_audio
    del audio_2

    # Pitching.
    pitch_audio = pitching(data, sampling_rate, random=True)
    audio_3 = feature_extraction(pitch_audio, sampling_rate)
    audio = np.vstack((audio, audio_3))
    del pitch_audio
    del audio_3

    # Pitching and Inject Noise.
    pitch_audio_1 = pitching(data, sampling_rate, random=True)
    pitch_noise_audio = inject_noise(pitch_audio_1, random=True)
    audio_4 = feature_extraction(pitch_noise_audio, sampling_rate)
    audio = np.vstack((audio, audio_4))
    del pitch_noise_audio
    del audio_4

    audio_features = audio

    del audio
    del data
    del sampling_rate
    gc.collect()

    return audio_features


# Increase ndarray dimensions to [4,2376].
def increase_ndarray_size(features_test):
    tmp = np.zeros([4, 2377])
    offsets = [0, 1]
    insert_here = tuple(
        [
            slice(offsets[dim], offsets[dim] + features_test.shape[dim])
            for dim in range(features_test.ndim)
        ]
    )

    tmp[insert_here] = features_test
    features_test = tmp

    del tmp
    gc.collect()

    features_test = np.delete(features_test, 0, axis=1)

    return features_test


# Determine if ndarray needs to be increase in size.
def increase_array_size(audio_features):
    if audio_features.shape[1] < 2376:
        audio_features = increase_ndarray_size(audio_features)
    return audio_features


@profile
def predict(audio_features):
    if "model_path" not in st.session_state:
        st.session_state["model_path"] = st.secrets["model_path"]

    # if "X_train_path" not in st.session_state:
    #     st.session_state["X_train_path"] = st.secrets["X_train_path"]

    # if "X_valid_path" not in st.session_state:
    #     st.session_state["X_valid_path"] = st.secrets["X_valid_path"]

    # if "X_test_path" not in st.session_state:
    #     st.session_state["X_test_path"] = st.secrets["X_test_path"]

    # if (
    #     ("X_train" not in st.session_state)
    #     or ("X_valid" not in st.session_state)
    #     or ("X_test" not in st.session_state)
    # ):
    #     if "X_train" in st.session_state:
    #         del st.session_state["X_train"]
    #     if "X_valid" in st.session_state:
    #         del st.session_state["X_valid"]
    #     if "X_test" in st.session_state:
    #         del st.session_state["X_test"]

    #     gc.collect()

    # if "is_download" not in st.session_state:
    # st.session_state["client"].download_file(
    #     st.session_state["bucket_name"],
    #     st.session_state["X_train_path"],
    #     "X_train.parquet",
    # )

    # st.session_state["client"].download_file(
    #     st.session_state["bucket_name"],
    #     st.session_state["X_valid_path"],
    #     "X_valid.parquet",
    # )

    # st.session_state["client"].download_file(
    #     st.session_state["bucket_name"],
    #     st.session_state["X_test_path"],
    #     "X_test.parquet",
    # )

    # st.session_state["is_download"] = True

    # We have X which is data augmentation + data extraction.
    # X_train = pd.read_parquet("X_train.parquet")
    # st.session_state["X_train"] = X_train
    # del X_train

    # X_valid = pd.read_parquet("X_valid.parquet")
    # st.session_state["X_valid"] = X_valid
    # del X_valid

    # X_test = pd.read_parquet("X_test.parquet")
    # st.session_state["X_test"] = X_test
    # del X_test
    # gc.collect()

    if "standard_scaler" not in st.session_state:
        st.session_state["client"].download_file(
            st.session_state["bucket_name"],
            st.session_state["standard_scaler_path"],
            "standard_scaler.save",
        )

        st.session_state["standard_scaler"] = joblib.load("standard_scaler.save")

    # st.session_state["standard_scaler"].fit_transform(
    #     st.session_state["X_train"].values
    # )
    # del st.session_state["X_train"]
    # st.session_state["standard_scaler"].transform(st.session_state["X_valid"].values)
    # del st.session_state["X_valid"]
    # st.session_state["standard_scaler"].transform(st.session_state["X_test"].values)
    # del st.session_state["X_test"]

    audio_features = st.session_state["standard_scaler"].transform(audio_features)
    audio_features = np.expand_dims(audio_features, axis=2)

    del st.session_state["standard_scaler"]
    gc.collect()

    if "model" not in st.session_state:
        st.session_state["client"].download_file(
            st.session_state["bucket_name"],
            st.session_state["model_path"],
            "model.h5",
        )

        # Load the model.
        st.session_state["model"] = load_model("model.h5")

    st.session_state["y_pred"] = list(st.session_state["model"].predict(audio_features))

    if "y_pred" not in st.session_state:
        st.session_state["y_pred"] = list(
            st.session_state["model"].predict(audio_features)
        )

    del audio_features
    gc.collect()

    st.session_state["y_pred"] = list(np.argmax(st.session_state["y_pred"], axis=1))

    if "emotions_classes" not in st.session_state:
        st.session_state["emotions_classes"] = [
            "angry",
            "calm",
            "disgust",
            "fear",
            "happy",
            "neutral",
            "sad",
            "surprise",
        ]

    try:
        st.session_state["pred_emotion"] = st.session_state["emotions_classes"][
            mode(st.session_state["y_pred"])
        ]
        del st.session_state["emotions_classes"]
        del st.session_state["y_pred"]
        gc.collect()
    except:
        st.session_state["pred_emotion"] = st.session_state["emotions_classes"][
            st.session_state["y_pred"][0]
        ]
        del st.session_state["emotions_classes"]
        del st.session_state["y_pred"]
        gc.collect()


# Use local CSS.
def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)


# Load local CSS.
local_css("styles/style.css")

# Bootstrap cards with reference to CSS.
st.markdown(
    """<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/bootstrap@5.2.2/dist/css/bootstrap.min.css"
        integrity="sha384-Zenh87qX5JnK2Jl0vWa8Ck2rdkQ2Bzep5IDxbcnCeuOxjzrPF/et3URy9Bv1WTRi" crossorigin="anonymous">
    """,
    unsafe_allow_html=True,
)

# Title.
if "title" not in st.session_state:
    st.session_state[
        "title"
    ] = f"""<p align="center" style="font-family: monospace; color: #FAF9F6; font-size: 2.3rem;"> Voice Emotion Recognition on Audio</p>"""
st.markdown(st.session_state["title"], unsafe_allow_html=True)

# Image.
if "image" not in st.session_state:
    st.session_state[
        "image"
    ] = "https://t4.ftcdn.net/jpg/03/27/36/95/360_F_327369570_CAxxxHHLvjk6IJ3wGi1kuW6WTtqjaMpc.jpg"
st.image(st.session_state["image"], use_column_width=True)

# Header.
if "header" not in st.session_state:
    st.session_state[
        "header"
    ] = f"""<p align="center" style="font-family: monospace; color: #FAF9F6; font-size: 1.7rem;"> Click to generate a random prompt and emotion:</p>"""
st.markdown(st.session_state["header"], unsafe_allow_html=True)


# Emotion emoji animation.
def styling(particle):
    return st.markdown(
        f"""
        <div class="snowflake">{particle}</div>
        <div class="snowflake">{particle}</div>
        <div class="snowflake">{particle}</div>
        <div class="snowflake">{particle}</div>
        <div class="snowflake">{particle}</div>
        """,
        unsafe_allow_html=True,
    )


# Create the grid.
def make_grid(rows, cols):
    grid = [0] * rows
    for i in range(rows):
        with st.container():
            grid[i] = st.columns(cols)
    return grid


# Prompt button.
def prompt_btn():
    if not (st.session_state["is_first_time_prompt"]):
        styling(particle=st.session_state["particle"])

    prompt = '"' + random.choice(st.session_state["prompts"]) + '"'
    st.session_state["prompt"] = prompt

    st.markdown(
        f"""
        <p align="center" style="font-family: monospace; color: #ffffff; font-size: 2rem;">
        {st.session_state["prompt"]}</p>
        """,
        unsafe_allow_html=True,
    )

    if not (st.session_state["is_first_time_prompt"]) and (
        st.session_state["emotion_dict"].get(st.session_state["emotion"]) != None
    ):
        st.markdown(
            f"""
            <p align="center" style="font-family: monospace; color: #FAF9F6; font-size: 2.5rem;">
            Try to sound {st.session_state["emotion_dict"].get(st.session_state["emotion"])}</p>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f"""
            <p align="center" style="font-family: monospace; color: #FAF9F6; font-size: 2.5rem;">
            Please generate an emotion!</p>
            """,
            unsafe_allow_html=True,
        )


# Emotion button.
def emotion_btn():
    st.session_state["initial_styling"] = False
    st.session_state["is_first_time_prompt"] = False

    emotion = random.choice(list(st.session_state["emotion_dict"]))
    partition = st.session_state["emotion_dict"].get(emotion).split(" ")
    emotion = partition[0]
    st.session_state["emotion"] = emotion

    if st.session_state["emotion"] == "disgusted":
        st.session_state["emotion"] = "disgust"

    if st.session_state["emotion"] == "scared":
        st.session_state["emotion"] = "fear"

    if st.session_state["emotion"] == "surprised":
        st.session_state["emotion"] = "surprise"

    particle = partition[1]
    st.session_state["particle"] = particle
    styling(particle=st.session_state["particle"])

    st.markdown(
        f"""
        <p align="center" style="font-family: monospace; color: #ffffff; font-size: 2rem;">
        {st.session_state["prompt"]}</p>
        """,
        unsafe_allow_html=True,
    )
    st.markdown(
        f"""
        <p align="center" style="font-family: monospace; color: #FAF9F6; font-size: 2.5rem;">
        Try to sound {st.session_state["emotion_dict"].get(st.session_state["emotion"])}</p>
        """,
        unsafe_allow_html=True,
    )


# Connect to te server and save the frames from the audio receiver.
def save_frames_from_audio_receiver():
    webrtc_ctx = webrtc_streamer(
        key="sendonly-audio",
        mode=WebRtcMode.SENDONLY,
        audio_receiver_size=1024,
        # Uncomment for deployment.
        rtc_configuration=st.session_state["RTC_CONFIGURATION"],
        media_stream_constraints={"video": False, "audio": True},
    )
    if "audio_buffer" not in st.session_state:
        st.session_state["audio_buffer"] = pydub.AudioSegment.empty()

    if "audio_buffer_bytes" not in st.session_state:
        st.session_state["audio_buffer_bytes"] = b""

    if "valid_audio" not in st.session_state:
        st.session_state["valid_audio"] = False

    success_indicator = st.empty()
    first_indicator = st.empty()
    second_indicator = st.empty()

    sound_window_len = 3000
    sound_window_buffer = None

    while True:
        if webrtc_ctx.audio_receiver:
            try:
                audio_frames = webrtc_ctx.audio_receiver.get_frames(timeout=1)
            except queue.Empty:
                time.sleep(0.5)
                break

            first_indicator.write(
                f"""
                <p align="center" style="font-family: monospace; color: #ffffff; font-size: 2rem;">
                Recording has started🏁</p>
                """,
                unsafe_allow_html=True,
            )

            second_indicator.write(
                f"""
                <p align="center" style="font-family: monospace; color: #ffffff; font-size: 2rem;">
                Press STOP to stop the recording.</p>
                """,
                unsafe_allow_html=True,
            )

            for audio_frame in audio_frames:
                sound = pydub.AudioSegment(
                    data=audio_frame.to_ndarray().tobytes(),
                    sample_width=audio_frame.format.bytes,
                    frame_rate=audio_frame.sample_rate,
                    channels=len(audio_frame.layout.channels),
                )
                st.session_state["audio_buffer"] += sound
                st.session_state["audio_buffer_bytes"] += sound.raw_data
                st.session_state["valid_audio"] = True

            if len(st.session_state["audio_buffer"]) > 0:
                if sound_window_buffer is None:
                    sound_window_buffer = pydub.AudioSegment.silent(
                        duration=sound_window_len
                    )

                sound_window_buffer += st.session_state["audio_buffer"]

                if len(sound_window_buffer) > sound_window_len:
                    sound_window_buffer = sound_window_buffer[-sound_window_len:]

            if sound_window_buffer:
                sound_window_buffer = sound_window_buffer.set_channels(1)
        else:
            break

    first_indicator.write(
        f"""
            <p align="center" style="font-family: monospace; color: #ffffff; font-size: 1.5rem;">NOTE📝</p>
            <p align="center" style="font-family: monospace; color: #ffffff; font-size: 1.3rem;">
            Press START and wait for a minute to connect to the server. In the meantime get your voice ready!</p>
            <p align="center" style="font-family: monospace; color: #ffffff; font-size: 1.4rem;">
            Please be patient as this utilizes a free public server. The recording must be about 3 seconds.</p>
            <p align="center" style="font-family: monospace; color: #ffffff; font-size: 0.9rem;">
            If for any reason the recording did not start after a minute, then press the STOP button and start the recording again as there was a server connection error.</p>
        """,
        unsafe_allow_html=True,
    )

    audio_buffer_bytes = st.session_state["audio_buffer_bytes"]
    container = BytesIO()

    wf = wave.open(container, "wb")
    wf.setnchannels(2)
    wf.setsampwidth(2)
    wf.setframerate(44100)
    wf.writeframes(audio_buffer_bytes)
    wf.close()

    container.seek(0)

    data_package = container.read()

    bytes = BytesIO(data_package)

    del wf
    del data_package

    st.session_state["recording_name"] = (
        st.session_state["recordings_path"] + st.session_state["dt_string"] + ".wav"
    )
    if (st.session_state["valid_audio"] == True) and (bytes.getbuffer().nbytes > 100):
        success_indicator.write(
            f"""<p align="center" style="font-family: monospace; color: #ffffff; font-size: 1.5rem;">✅Recording successful!</p>""",
            unsafe_allow_html=True,
        )
        st.session_state["client"].upload_fileobj(
            bytes,
            st.session_state["bucket_name"],
            st.session_state["recording_name"],
        )

        st.session_state["audio_buffer"] = pydub.AudioSegment.empty()
        st.session_state["audio_buffer_bytes"] = b""
    del bytes
    gc.collect()


# Recording button.
def recording():
    styling(particle=st.session_state["particle"])
    st.markdown(
        f"""
        <p align="center" style="font-family: monospace; color: #FAF9F6; font-size: 2rem;">
        {st.session_state["prompt"]}</p>
        """,
        unsafe_allow_html=True,
    )
    if not (st.session_state["is_first_time_prompt"]):
        if st.session_state["emotion_dict"].get(st.session_state["emotion"]) == None:
            st.markdown(
                f"""
                <p align="center" style="font-family: monospace; color: #FAF9F6; font-size: 2.5rem;">
                Please generate an emotion. </p>
                """,
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f"""
                <p align="center" style="font-family: monospace; color: #FAF9F6; font-size: 2.5rem;">
                Try to sound {st.session_state["emotion_dict"].get(st.session_state["emotion"])}</p>
                """,
                unsafe_allow_html=True,
            )
    else:
        st.markdown(
            f"""
            <p align="center" style="font-family: monospace; color: #FAF9F6; font-size: 2.5rem;">
            Please generate a prompt and an emotion!</p>
            <p align="center" style="font-family: monospace; color: #FAF9F6; font-size: 2.5rem;">Then, record your audio.</p>
            """,
            unsafe_allow_html=True,
        )
    st.session_state["webrtc"] = False
    app_dict = {"VERA": save_frames_from_audio_receiver}
    app_func = app_dict["VERA"]
    app_func()


# Play button.
@profile
def play_btn():
    styling(particle=st.session_state["particle"])
    st.markdown(
        f"""
        <p align="center" style="font-family: monospace; color: #FAF9F6; font-size: 2rem;">
        {st.session_state["prompt"]}</p>
        """,
        unsafe_allow_html=True,
    )
    if not (st.session_state["is_first_time_prompt"]):
        if st.session_state["emotion_dict"].get(st.session_state["emotion"]) == None:
            st.markdown(
                f"""
                <p align="center" style="font-family: monospace; color: #FAF9F6; font-size: 2.5rem;">
                Please generate an emotion. </p>
                """,
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f"""
                <p align="center" style="font-family: monospace; color: #FAF9F6; font-size: 2.5rem;">
                Try to sound {st.session_state["emotion_dict"].get(st.session_state["emotion"])}</p>
                """,
                unsafe_allow_html=True,
            )
    else:
        st.markdown(
            f"""
            <p align="center" style="font-family: monospace; color: #FAF9F6; font-size: 2.5rem;">
            Please generate a prompt and an emotion!</p>
            <p align="center" style="font-family: monospace; color: #FAF9F6; font-size: 2.5rem;">Then, record your audio.</p>
            """,
            unsafe_allow_html=True,
        )
    try:
        recording_name = st.session_state["recording_name"]
        if "audio_bytes" not in st.session_state:
            st.session_state["client"].download_file(
                st.session_state["bucket_name"], recording_name, "recording_name"
            )
            audio_file = open("recording_name", "rb")
            audio_bytes = audio_file.read()
            st.session_state["audio_bytes"] = audio_bytes
            st.audio(audio_bytes)
        else:
            st.audio(st.session_state["audio_bytes"])

        del audio_file
        del audio_bytes
        gc.collect()
    except:
        st.info(
            "Please record sound first. If for any reason you have recorded and it does not display, then click on the Record button again to see if the recording has been succeeded.",
            icon="ℹ️",
        )


# Classify button.
@profile
def classify_btn():
    st.session_state["calculated_time"] = False
    try:
        recording_name = st.session_state["recording_name"]

        st.session_state["client"].download_file(
            st.session_state["bucket_name"],
            recording_name,
            "recording_name",
        )
        print("test1")
        audio_features = get_features("recording_name")
        print("test2")
        audio_features = increase_array_size(audio_features)
        print("test3")
        predict(audio_features)
        print("test4")

        del audio_features
        gc.collect()

        if st.session_state["pred_emotion"] == "disgust":
            st.session_state["pred_emotion"] = "disgusted"

        if st.session_state["pred_emotion"] == "fear":
            st.session_state["pred_emotion"] = "scared"

        if st.session_state["pred_emotion"] == "surprise":
            st.session_state["pred_emotion"] = "surprised"

        if st.session_state["emotion"] == "disgust":
            st.session_state["emotion"] = "disgusted"

        if st.session_state["emotion"] == "fear":
            st.session_state["emotion"] = "scared"

        if st.session_state["emotion"] == "surprise":
            st.session_state["emotion"] = "surprised"

        print("test5")

        if st.session_state["emotion"] != "":
            print("test6")
            if st.session_state["pred_emotion"] in st.session_state["emotion"]:
                st.session_state["particle"] = "😆"
                styling(particle=st.session_state["particle"])
                st.markdown(
                    f"""
                    <p align="center" style="font-family: monospace; color: #FAF9F6; font-size: 1rem;">
                    Please refresh the webpage before attempting to record/classify a new audio.🔄</p>
                    <p align="center" style="font-family: monospace; color: #FAF9F6; font-size: 2.5rem;">
                    You tried to sound {st.session_state["emotion"].upper()} and you sounded {st.session_state["pred_emotion"].upper()}</p>
                    <p align="center" style="font-family: monospace; color: #FAF9F6; font-size: 2.5rem;">Well done!👍</p>
                    """,
                    unsafe_allow_html=True,
                )

                try:
                    if "audio_bytes" not in st.session_state:
                        st.session_state["client"].download_file(
                            st.session_state["bucket_name"],
                            recording_name,
                            "recording_name",
                        )
                        audio_file = open("recording_name", "rb")
                        audio_bytes = audio_file.read()
                        st.session_state["audio_bytes"] = audio_bytes
                        st.audio(st.session_state["audio_bytes"])
                    else:
                        st.audio(st.session_state["audio_bytes"])

                    st.session_state["recording_name"] = (
                        st.session_state["recordings_path"]
                        + st.session_state["dt_string"]
                        + "_"
                        + st.session_state["pred_emotion"]
                        + ".wav"
                    )

                    bytes = BytesIO(st.session_state["audio_bytes"])
                    st.session_state["client"].upload_fileobj(
                        bytes,
                        st.session_state["bucket_name"],
                        st.session_state["recording_name"],
                    )

                    del st.session_state["audio_bytes"]
                    del bytes
                    gc.collect()

                    st.balloons()
                except Exception as e:
                    print(e)
                    st.error(
                        "Something went wrong when loading the audio. Please try again."
                    )
            else:
                st.session_state["particle"] = "😢"
                styling(particle=st.session_state["particle"])
                st.markdown(
                    f"""
                    <p align="center" style="font-family: monospace; color: #FAF9F6; font-size: 1rem;">
                    Please refresh the webpage before attempting to record/classify a new audio.🔄</p>
                    <p align="center" style="font-family: monospace; color: #FAF9F6; font-size: 2.5rem;">
                    You tried to sound {st.session_state["emotion"].upper()} however you sounded {st.session_state["pred_emotion"].upper()}👎</p>
                    """,
                    unsafe_allow_html=True,
                )

                try:
                    if "audio_bytes" not in st.session_state:
                        st.session_state["client"].download_file(
                            st.session_state["bucket_name"],
                            recording_name,
                            "recording_name",
                        )
                        audio_file = open("recording_name", "rb")
                        audio_bytes = audio_file.read()
                        st.session_state["audio_bytes"] = audio_bytes
                        st.audio(st.session_state["audio_bytes"])
                    else:
                        st.audio(st.session_state["audio_bytes"])

                    st.session_state["recording_name"] = (
                        st.session_state["recordings_path"]
                        + st.session_state["dt_string"]
                        + "_"
                        + st.session_state["pred_emotion"]
                        + ".wav"
                    )

                    bytes = BytesIO(st.session_state["audio_bytes"])
                    st.session_state["client"].upload_fileobj(
                        bytes,
                        st.session_state["bucket_name"],
                        st.session_state["recording_name"],
                    )

                    del st.session_state["audio_bytes"]
                    del bytes
                    gc.collect()
                except Exception as e:
                    print(e)
                    st.error(
                        "Something went wrong when loading the audio. Please try again."
                    )
        else:
            print("test7")
            st.markdown(
                f"""
                <p align="center" style="font-family: monospace; color: #FAF9F6; font-size: 2.5rem;"> Please generate a prompt and an emotion.</p>
                    """,
                unsafe_allow_html=True,
            )
        st.session_state["recording_name"] = (
            st.session_state["recordings_path"] + st.session_state["dt_string"] + ".wav"
        )
    except:
        st.info("Please record sound first.", icon="ℹ️")


# User Interface.
if st.session_state["initial_styling"]:
    styling(particle=st.session_state["particle"])

# Create the custom grid.
grid1 = make_grid(3, (12, 12, 4))

# Prompt Button.
prompt = grid1[0][0].button("Prompt")
if prompt or st.session_state["is_prompt"]:
    st.session_state["webrtc"] = False
    st.session_state["is_emotion"] = False
    prompt_btn()

# Emotion Button.
emotion = grid1[0][2].button("Emotion")
if emotion or st.session_state["is_emotion"]:
    st.session_state["webrtc"] = False
    st.session_state["is_prompt"] = False
    emotion_btn()

# Create the custom grid.
grid2 = make_grid(3, (12, 12, 3.8))

# Play Button.
play = grid2[0][0].button("Play")
if play:
    st.session_state["webrtc"] = False
    play_btn()

# Classify Button.
classify = grid2[0][2].button("Classify")
if classify:
    with st.spinner("Classifying your recording!🔬"):
        classify_indicator = st.empty()
        classify_indicator.markdown(
            f"""<p align="justify" style="font-family: monospace; color: #FAF9F6; font-size: 0.7rem;">
        &emsp; Speech Emotion Recognition (SER) is the task of recognizing the emotional aspects of \
        speech irrespective of the semantic contents. While humans can efficiently perform \
        this task as a natural part of speech communication, the ability to conduct it \
        automatically using programmable devices is still an ongoing subject of research. \
        Studies of automatic emotion recognition systems aim to create efficient, \
        real-time methods of detecting the emotions of mobile phone users, \
        call center operators and customers, car drivers, pilots, and many other \
        human-machine communication users. Adding emotions to machines has been recognized as \
        a critical factor in making machines appear and act in a human-like manner \
        (André et al., 2004).</p>
        <p align="justify" style="font-family: monospace; color: #FAF9F6; font-size: 0.7rem;">
        &emsp; If you've ever communicated with another human being-and we \
        hope you have-you know that even with all of our experience with social interactions, \
        it can still be tricky to determine someone's emotional state just from talking to them.\
        This is doubly problematic for attempts at emotion recognition. Not only are you trying \
        to make a system to do something that's tricky for humans, you're going to do so using a \
        dataset that humans have labeled based on the emotion that they think is present, even \
        though they might not agree, and even though their labels might not accurately match the \
        emotion the speaker was actually feeling.This is further complicated by the fact that the \
        audio used to train the model might be acted data, and not people actually expressing the \
        emotion that they're experiencing. Plus, most emotion recognition systems only look at audio \
        data, and don't include other things that could help make a determination, such as body \
        language or facial expressions (Deepgram, 2022).</p>""",
            unsafe_allow_html=True,
        )
        classify_btn()
        classify_indicator.empty()

if "webrtc" not in st.session_state:
    st.session_state["webrtc"] = False

st.session_state["webrtc"] = False

# Create the custom grid.
grid3 = make_grid(3, (12, 12, 4))

# Record Button.
record = grid3[0][0].button("Record")
if record:
    st.session_state["record"] = True
    if st.session_state["webrtc"] == False:
        st.markdown(
            f"""
                <p align="center" style="font-family: monospace; color: #FAF9F6; font-size: 2rem;">
                Please generate a prompt and an emotion!</p>
                <p align="center" style="font-family: monospace; color: #FAF9F6; font-size: 2rem;">Then, record your audio.</p>
                """,
            unsafe_allow_html=True,
        )
if (
    (st.session_state["prompt"] != "")
    and (st.session_state["emotion"] != "")
    and play == False
    and classify == False
    and prompt == False
    and emotion == False
    and (st.session_state["record"] == True)
):
    st.session_state["webrtc"] == False
    recording()

# GitHub repository of project.
st.markdown(
    f"""
        <p align="center" style="font-family: monospace; color: #FAF9F6; font-size: 1rem;"><b> Check out our
        <a href="https://github.com/GeorgiosIoannouCoder/vera-deployed" style="color: #FAF9F6;"> GitHub repository</a></b>
        </p>
""",
    unsafe_allow_html=True,
)


for name in dir():
    if not name.startswith("_"):
        del globals()[name]

import gc

gc.collect()
