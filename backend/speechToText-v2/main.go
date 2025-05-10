package main

import (
	"context"
	"fmt"
	"log"
	"os"

	speech "cloud.google.com/go/speech/apiv1"
	speechpb "google.golang.org/genproto/googleapis/cloud/speech/v1"
)

func main() {
	// Set up Google Cloud Speech-to-Text client
	ctx := context.Background()
	client, err := speech.NewClient(ctx)
	if err != nil {
		log.Fatalf("Failed to create client: %v", err)
	}
	defer client.Close()

	// Open the audio file
	audioFile := "path/to/your/audiofile.wav"
	file, err := os.Open(audioFile)
	if err != nil {
		log.Fatalf("Failed to open audio file: %v", err)
	}
	defer file.Close()

	// Read the audio file
	data, err := os.ReadFile(audioFile)
	if err != nil {
		log.Fatalf("Failed to read audio file: %v", err)
	}

	// Configure the request
	req := &speechpb.RecognizeRequest{
		Config: &speechpb.RecognitionConfig{
			Encoding:        speechpb.RecognitionConfig_LINEAR16,
			SampleRateHertz: 16000,
			LanguageCode:    "en-US",
		},
		Audio: &speechpb.RecognitionAudio{
			AudioSource: &speechpb.RecognitionAudio_Content{
				Content: data,
			},
		},
	}

	// Perform the speech recognition request
	resp, err := client.Recognize(ctx, req)
	if err != nil {
		log.Fatalf("Failed to recognize: %v", err)
	}

	// Print the results
	for _, result := range resp.Results {
		for _, alt := range result.Alternatives {
			fmt.Printf("Transcript: %v\n", alt.Transcript)
		}
	}
}

// Save the results to a text file
outputFile := "path/to/your/outputfile.txt"
out, err := os.Create(outputFile)
if err != nil {
	log.Fatalf("Failed to create output file: %v", err)
}
defer out.Close()

for _, result := range resp.Results {
	for _, alt := range result.Alternatives {
		_, err := out.WriteString(fmt.Sprintf("Transcript: %v\n", alt.Transcript))
		if err != nil {
			log.Fatalf("Failed to write to output file: %v", err)
		}
	}
}

fmt.Printf("Transcription saved to %s\n", outputFile)