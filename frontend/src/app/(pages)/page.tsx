"use client";

import Link from "next/link";
import React, { useEffect, useRef, useState } from "react";
import { useRouter } from "next/navigation";

const SignMeetWorkplace: React.FC = () => {
  const router = useRouter();
  const audioRef = useRef<HTMLAudioElement | null>(null);
  const [isListening, setIsListening] = useState(false);
  const [recognitionInstance, setRecognitionInstance] = useState<any>(null);
  const [status, setStatus] = useState<"idle" | "playing" | "listening">(
    "idle"
  );

  // Function to start speech recognition
  const startSpeechRecognition = () => {
    // Sử dụng any type để tránh các lỗi TypeScript
    const SpeechRecognition =
      (window as any).SpeechRecognition ||
      (window as any).webkitSpeechRecognition;

    if (!SpeechRecognition) {
      console.error("Trình duyệt không hỗ trợ nhận diện giọng nói");
      return null;
    }

    const recognition = new SpeechRecognition();

    recognition.lang = "vi-VN";
    recognition.continuous = true;
    recognition.interimResults = false;

    recognition.onresult = (event: any) => {
      const transcript = event.results[event.results.length - 1][0].transcript;
      const command = transcript.trim().toLowerCase();

      console.log("Nhận diện được:", command);

      // Stop listening after receiving a command
      recognition.stop();
      setIsListening(false);
      setStatus("idle");

      // Navigate based on the command
      if (command.includes("đăng nhập")) {
        router.push("/signin");
      } else if (command.includes("đăng ký")) {
        router.push("/signup");
      } else if (
        command.includes("vào cuộc họp") ||
        command.includes("tham gia cuộc họp")
      ) {
        router.push("/meeting");
      }
    };

    recognition.onstart = () => {
      console.log("Bắt đầu nhận diện giọng nói");
      setIsListening(true);
      setStatus("listening");
    };

    recognition.onend = () => {
      console.log("Kết thúc nhận diện giọng nói");
      if (status === "listening") {
        setIsListening(false);
        setStatus("idle");
      }
    };

    recognition.onerror = (event: any) => {
      if (event.error === "no-speech") {
        console.error("Không nhận diện được giọng nói");
      }
      setIsListening(false);
      setStatus("idle");
    };

    recognition.start();
    setRecognitionInstance(recognition);
    return recognition;
  };

  // Function to start the welcome process
  const startWelcomeProcess = () => {
    // Don't start if already playing or listening
    if (status !== "idle") return;

    // Create welcome audio element if it doesn't exist
    if (!audioRef.current) {
      audioRef.current = new Audio("/audio/welcome_dashboard.mp3");
    }

    // Set status to playing
    setStatus("playing");

    // Play welcome audio and then start speech recognition
    audioRef.current.onended = () => {
      startSpeechRecognition();
    };

    audioRef.current.play().catch((err) => {
      console.error("Không thể phát âm thanh:", err);
      // Start speech recognition anyway if audio fails
      startSpeechRecognition();
    });
  };

  // Add keyboard event listener for Space key
  useEffect(() => {
    const handleKeyDown = (event: KeyboardEvent) => {
      // Check if Space key is pressed and not in a text input or textarea
      if (
        event.code === "Space" &&
        status === "idle" &&
        !(
          event.target instanceof HTMLInputElement ||
          event.target instanceof HTMLTextAreaElement
        )
      ) {
        event.preventDefault(); // Prevent page scrolling
        startWelcomeProcess();
      }
    };

    // Add event listener
    window.addEventListener("keydown", handleKeyDown);

    // Remove event listener on cleanup
    return () => {
      window.removeEventListener("keydown", handleKeyDown);
    };
  }, [status]);

  // Cleanup function for audio and speech recognition
  useEffect(() => {
    return () => {
      if (audioRef.current) {
        audioRef.current.pause();
        audioRef.current.onended = null;
      }

      if (recognitionInstance) {
        recognitionInstance.stop();
      }
    };
  }, [recognitionInstance]);

  // Get button text and color based on status
  const getButtonConfig = () => {
    switch (status) {
      case "playing":
        return {
          text: "Đang phát âm thanh...",
          className: "bg-blue-500",
        };
      case "listening":
        return {
          text: "Đang nghe...",
          className: "bg-red-500",
        };
      default:
        return {
          text: "Trợ lý",
          className: "bg-green-600",
        };
    }
  };

  const buttonConfig = getButtonConfig();

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-600 via-blue-500 to-blue-700 flex flex-col items-center justify-center p-4 bg-pattern">
      {/* Logo and Title Section */}
      <div className="mb-20 text-center animate-fade-down">
        <div className="flex items-center justify-center mb-6">
          <svg
            xmlns="http://www.w3.org/2000/svg"
            className="h-16 w-16 text-white mr-3 animate-bounce"
            fill="none"
            viewBox="0 0 24 24"
            stroke="currentColor"
          >
            <path
              strokeLinecap="round"
              strokeLinejoin="round"
              strokeWidth={2}
              d="M7 8h10M7 12h4m1 8l-4-4H5a2 2 0 01-2-2V6a2 2 0 012-2h14a2 2 0 012 2v8a2 2 0 01-2 2h-3l-4 4z"
            />
          </svg>
          <svg
            xmlns="http://www.w3.org/2000/svg"
            className="h-16 w-16 text-white animate-bounce"
            fill="none"
            viewBox="0 0 24 24"
            stroke="currentColor"
          >
            <path
              strokeLinecap="round"
              strokeLinejoin="round"
              strokeWidth={2}
              d="M15 10l4.553-2.276A1 1 0 0121 8.618v6.764a1 1 0 01-1.447.894L15 14M5 18h8a2 2 0 002-2V8a2 2 0 00-2-2H5a2 2 0 00-2 2v8a2 2 0 002 2z"
            />
          </svg>
        </div>
        <h1 className="text-white text-5xl font-bold mb-3 tracking-tight">
          SignMeet
        </h1>
        <h2 className="text-white/90 text-2xl font-light">
          Video Call & Sign Language Support
        </h2>
        <div className="mt-4 text-white/80 text-sm">
          Nhấn phím <kbd className="bg-white/20 px-2 py-1 rounded">Space</kbd>{" "}
          hoặc nhấn vào nút trợ lý để bắt đầu
        </div>
      </div>

      {/* Main Container */}
      <div className="bg-white/80 backdrop-blur-lg rounded-2xl p-8 w-full max-w-md shadow-2xl animate-fade-up border border-white/20">
        {/* Join Meeting Button */}
        <Link href={"/meeting"}>
          <button
            className="w-full bg-blue-600 text-white py-4 px-6 rounded-lg mb-5 
                         hover:bg-blue-700 transition-all duration-300 font-semibold flex items-center justify-center
                         transform hover:scale-102 hover:shadow-lg active:scale-98"
          >
            <svg
              xmlns="http://www.w3.org/2000/svg"
              className="h-6 w-6 mr-2"
              fill="none"
              viewBox="0 0 24 24"
              stroke="currentColor"
            >
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                strokeWidth={2}
                d="M15 10l4.553-2.276A1 1 0 0121 8.618v6.764a1 1 0 01-1.447.894L15 14M5 18h8a2 2 0 002-2V8a2 2 0 00-2-2H5a2 2 0 00-2 2v8a2 2 0 002 2z"
              />
            </svg>
            Join a meeting
          </button>
        </Link>

        {/* Sign Up Button */}
        <Link href={"/signup"}>
          <button
            className="w-full bg-white text-blue-600 py-4 px-6 rounded-lg mb-4
                         border-2 border-blue-600/20 hover:border-blue-600 hover:bg-blue-50 
                         transition-all duration-300 font-semibold transform hover:scale-102 hover:shadow-md"
          >
            Sign up
          </button>
        </Link>

        {/* Sign In Button */}
        <Link href={"/signin"} prefetch>
          <button
            className="w-full bg-gray-100/80 text-gray-700 py-4 px-6 rounded-lg
                         hover:bg-gray-200 transition-all duration-300 font-medium 
                         transform hover:scale-102 hover:shadow-md"
          >
            Sign in
          </button>
        </Link>
      </div>

      {/* Floating Voice Assistant Button */}
      <button
        onClick={startWelcomeProcess}
        disabled={status !== "idle"}
        className={`fixed bottom-6 right-6 ${buttonConfig.className} text-white p-4 rounded-full
                   shadow-lg transition-all duration-300 flex items-center justify-center
                   hover:shadow-xl active:scale-95 disabled:opacity-70 disabled:cursor-not-allowed
                   z-50 w-16 h-16`}
        title={`${buttonConfig.text} (hoặc nhấn phím Space)`}
      >
        <svg
          xmlns="http://www.w3.org/2000/svg"
          className="h-8 w-8"
          fill="none"
          viewBox="0 0 24 24"
          stroke="currentColor"
        >
          <path
            strokeLinecap="round"
            strokeLinejoin="round"
            strokeWidth={2}
            d="M19 11a7 7 0 01-7 7m0 0a7 7 0 01-7-7m7 7v4m0 0H8m4 0h4m-4-8a3 3 0 01-3-3V5a3 3 0 116 0v6a3 3 0 01-3 3z"
          />
        </svg>

        {/* Status Indicator */}
        {status !== "idle" && (
          <span className="absolute top-0 right-0 h-4 w-4 rounded-full bg-white animate-pulse"></span>
        )}
      </button>
    </div>
  );
};

export default SignMeetWorkplace;
