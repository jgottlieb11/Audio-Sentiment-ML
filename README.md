# Audio Emotion Recognition and Speaker Diarization

This project aims to classify the emotions in audio files and perform speaker diarization to separate audio sources. The dataset consists of emotional speech data, and the project pipeline includes converting audio formats, organizing data, performing speaker diarization, augmenting the data, and classifying the emotions of the speakers. This project can be applied to understand the emotional state of individuals in a conversation, such as between a customer and an agent.


## Motivation

In many industries, including customer support and entertainment, understanding the emotional state of individuals can provide valuable insights. For example, detecting customer emotions during a support call helps identify how satisfied or frustrated they are, enabling the business to improve services accordingly. Emotion recognition can also be used in virtual assistants, therapeutic conversations, and other human-computer interactions.

## Dataset

* **RAVDESS Dataset**: The primary dataset for classification is the RAVDESS emotional speech dataset, which includes 24 professional actors (12 female, 12 male) vocalizing two lexically-matched statements in a neutral North American accent. Emotions include *neutral*, *calm*, *happy*, *sad*, *angry*, *fearful*, *disgust*, and *surprised*. The dataset can be found [here](https://www.kaggle.com/uwrfkaggler/ravdess-emotional-speech-audio).

## Process

This project is divided into several stages:

**Stage 1** - **Data Organization**: We use `extract_data.py` to organize the dataset by emotion. The audio files are moved into separate folders based on their emotion codes extracted from the file names.

**Stage 2** - **Speaker Diarization**: Using `audio_speaker_segmentation.py`, the audio files are split into separate chunks corresponding to different speakers. The goal is to isolate the speech segments from each speaker, enabling individual emotion classification.

**Stage 3** - **Data Augmentation**: To improve model robustness, we augment the audio files by adding noise, stretching the time, shifting the pitch, and applying noise reduction techniques. This process is handled by `data_augmentation.py`.

**Stage 4** - **Emotion Classification**: We apply machine learning models on the preprocessed audio data to classify emotions. This involves extracting MFCC features and training several models like MLP and XGBoost, as implemented in `emotion_classifier.py`.


## File Descriptions

- `extract_data.py`: Organizes the RAVDESS dataset by emotion, moving audio files into directories based on their emotion codes.
- `audio_speaker_segmentation.py`: Performs speaker diarization to separate and segment the audio into chunks corresponding to individual speakers.
- `data_augmentation.py`: Applies data augmentation techniques to improve the variety and robustness of the dataset by adding noise, changing pitch, and performing noise reduction.
- `emotion_classifier.py`: Contains the complete code to extract features, train the model, and classify emotions based on the RAVDESS dataset.

## Running the Project

To run this project, follow these steps:

1.. **Organize Data**: Use `extract_data.py` to organize the RAVDESS dataset into emotion-labeled folders.
3. **Speaker Diarization**: Use `audio_speaker_segmentation.py` to segment the audio based on speakers.
4. **Data Augmentation**: Optionally, run `data_augmentation.py` to enhance the dataset.
5. **Emotion Classification**: Run `emotion_classifier.py` to classify emotions using machine learning models.

## Conclusion

This project provides a robust pipeline for audio emotion recognition and speaker diarization. It is applicable to various industries where understanding human emotions is valuable, such as customer support, entertainment, and virtual assistants.

