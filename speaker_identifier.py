import os
import torch
import torchaudio
import numpy as np
from pyannote.audio import Pipeline, Model, Inference
from sklearn.cluster import DBSCAN
from sklearn.metrics.pairwise import cosine_distances
import warnings
from transformers import pipeline
warnings.filterwarnings("ignore")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"The device being used is {device}")

hf="****" #Enter huggingface token here

class SpeakerDiarization:
    def __init__(self, huggingface_token, embedding_model_path="pyannote/embedding"):
        self.hf_token = huggingface_token
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Initialize diarization pipeline
        self.diarization_pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1",
            use_auth_token=self.hf_token
        ).to(self.device)
        
        # Initialize embedding model
        self.embedding_model = Inference(
            Model.from_pretrained(embedding_model_path, use_auth_token=self.hf_token).to(self.device),
            window="whole"
        )
        
        # Initialize ASR model
        self.whisper_pipeline = pipeline(
            "automatic-speech-recognition", 
            model="openai/whisper-medium"
        )
        
        # Speaker mapping
        self.speaker_mapping = {}

    def extract_speaker_embeddings(self, audio_files, save_path="embeddings.npz"):
        all_embeddings = []
        speaker_info = []

        for audio_file in audio_files:
            print(f"\nProcessing file: {audio_file}")
            waveform, sample_rate = torchaudio.load(audio_file)
            waveform = waveform.to(self.device)

            # Perform diarization
            diarization = self.diarization_pipeline({"waveform": waveform, "sample_rate": sample_rate})

            for turn, _, speaker_label in diarization.itertracks(yield_label=True):
                start_time = turn.start
                end_time = turn.end
                
                # Skip very short segments
                if (end_time - start_time) < 1.0:
                    continue

                segment = waveform[:, int(start_time * sample_rate):int(end_time * sample_rate)].cpu()
                embedding = self.embedding_model({"waveform": segment, "sample_rate": sample_rate})
                embedding = embedding / np.linalg.norm(embedding)

                all_embeddings.append(embedding)
                speaker_info.append({
                    'audio_file': audio_file,
                    'start_time': start_time,
                    'end_time': end_time,
                    'local_speaker_label': speaker_label
                })

        # Save embeddings
        np.savez_compressed(save_path, 
                             embeddings=all_embeddings, 
                             speaker_info=speaker_info)
        
        return all_embeddings, speaker_info

    def cluster_speakers(self, embeddings, eps=0.5, min_samples=2):
        embeddings_array = np.vstack(embeddings)
        cosine_sim_matrix = 1 - cosine_distances(embeddings_array)
        distance_matrix = 1 - cosine_sim_matrix

        clustering = DBSCAN(
            eps=eps, 
            min_samples=min_samples, 
            metric='precomputed'
        )
        clustering.fit(distance_matrix)
        return clustering.labels_

    def assign_custom_labels(self, labels, speaker_mapping=None):
        cluster_to_name = {}
        for label in set(labels):
            if label == -1:
                continue
            
            if speaker_mapping and label in speaker_mapping:
                cluster_to_name[label] = speaker_mapping[label]
            else:
                cluster_to_name[label] = f"SPEAKER_{label:02d}"
        
        return cluster_to_name

    def match_new_speakers(self, new_embeddings, known_embeddings, known_labels, threshold=0.7):

        matched_labels = []
        for new_emb in new_embeddings:
            similarities = 1 - cosine_distances([new_emb], known_embeddings).flatten()
            max_sim_index = np.argmax(similarities)
            max_sim_value = similarities[max_sim_index]

            if max_sim_value > threshold:
                matched_labels.append(known_labels[max_sim_index])
            else:
                # Assign a new cluster label
                matched_labels.append(max(known_labels) + 1 if len(known_labels) > 0 else 0)

        return np.array(matched_labels)

    def process_known_speakers(self, audio_files, save_path="known_embeddings.npz", speaker_mapping=None):
        # Extract embeddings
        all_embeddings, speaker_info = self.extract_speaker_embeddings(audio_files, save_path=save_path)
        
        # Cluster speakers
        labels = self.cluster_speakers(all_embeddings)
        
        # Assign labels
        cluster_to_name = self.assign_custom_labels(labels, speaker_mapping)
        
        # Print speaker segments with global labels
        print("\nSpeaker Segments in Training Audio Files:")
        for idx, (embedding, info) in enumerate(zip(all_embeddings, speaker_info)):
            # Get the cluster label for this embedding
            cluster_label = labels[idx]
            
            # Skip noise points (cluster -1)
            if cluster_label == -1:
                continue
            
            # Get the global speaker name for this cluster
            speaker_name = cluster_to_name.get(cluster_label, f"SPEAKER_{cluster_label:02d}")
            
            print(f"File: {info['audio_file']}")
            print(f"  Segment: {info['start_time']:.2f}s - {info['end_time']:.2f}s")
            print(f"  Local Label: {info['local_speaker_label']}")
            print(f"  Global Label: {speaker_name} (Cluster {cluster_label})")
            print()
        
        # Save with labels
        np.savez_compressed(save_path, 
                            embeddings=all_embeddings, 
                            speaker_info=speaker_info, 
                            labels=labels,
                            cluster_to_name=cluster_to_name)
        
        print("Known speakers processed and saved.")
        return all_embeddings, labels, cluster_to_name

    def transcribe_with_labels(self, audio_file, known_embeddings_path="known_embeddings.npz", max_segment_duration=10):
        # Load known embeddings
        known_data = np.load(known_embeddings_path, allow_pickle=True)
        known_embeddings = known_data['embeddings']
        known_labels = known_data.get('labels', None)
        cluster_to_name = dict(known_data['cluster_to_name'].item()) if 'cluster_to_name' in known_data else {}

        # Load and process new audio
        waveform, sample_rate = torchaudio.load(audio_file)
        waveform = waveform.to(self.device)

        # Perform diarization
        diarization = self.diarization_pipeline({"waveform": waveform, "sample_rate": sample_rate})

        transcriptions = []

        for turn, _, speaker_label in diarization.itertracks(yield_label=True):
            start_time = turn.start
            end_time = turn.end
            duration = end_time - start_time

            # Skip segments shorter than 1 second
            if duration < 1.0:
                continue

            # Extract segment and embedding
            start_idx = int(start_time * sample_rate)
            end_idx = int(end_time * sample_rate)
            segment = waveform[:, start_idx:end_idx].cpu()
            embedding = self.embedding_model({"waveform": segment, "sample_rate": sample_rate})
            embedding = embedding / np.linalg.norm(embedding)

            # Match embedding with known speakers
            matched_labels = self.match_new_speakers([embedding], known_embeddings, known_labels)
            label = matched_labels[0]

            # Determine speaker name
            speaker_name = cluster_to_name.get(label, f"SPEAKER_{label:02d}")

            # Split segment for transcription
            num_chunks = int(np.ceil(duration / max_segment_duration))
            chunk_duration = int(sample_rate * max_segment_duration)

            for i in range(num_chunks):
                chunk_start = i * chunk_duration
                chunk_end = min((i + 1) * chunk_duration, segment.shape[1])
                chunk = segment[:, chunk_start:chunk_end]

                # Transcribe
                transcription = self.whisper_pipeline(chunk.squeeze().numpy())
                transcribed_text = transcription['text']

                transcriptions.append({
                    'speaker_name': speaker_name,
                    'start_time': start_time + (chunk_start / sample_rate),
                    'end_time': start_time + (chunk_end / sample_rate),
                    'text': transcribed_text
                })

        return transcriptions

def main():
    # HuggingFace token (replace with your token)
    HF_TOKEN = hf

    # Initialize speaker diarization
    speaker_diarizer = SpeakerDiarization(HF_TOKEN)
   # Known audio files for training
    known_audio_files = ["./data/sample1.wav", "./data/sample2.wav", "./data/sample3.wav","./data/sample5.wav","./data/sample6.wav","./data/trump1.wav","./data/john1.wav"]

    # Custom speaker mapping
    speaker_mapping = {
        0: "Vaibhav",
        1: "Sushmit",
        3:"S Jaishankar",
        4:"Donald Trump",
        5:"John",
    }

    # Process known speakers
    speaker_diarizer.process_known_speakers(
        known_audio_files, 
        save_path="known_embeddings.npz", 
        speaker_mapping=speaker_mapping
    )

    # New audio file to transcribe
    new_audio_file = "./data/john2.wav"
    
    # Transcribe with speaker labels
    results = speaker_diarizer.transcribe_with_labels(
        new_audio_file, 
        known_embeddings_path="known_embeddings.npz"
    )

    # Print results
    for segment in results:
        print(f"[{segment['speaker_name']}] {segment['text']} ({segment['start_time']:.2f}s - {segment['end_time']:.2f}s)")

if __name__ == "__main__":
    main()