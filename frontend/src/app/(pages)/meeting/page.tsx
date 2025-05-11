"use client";
import Link from "next/link";
import React, { useState } from "react";
import { useRouter } from "next/navigation";
// Input: nothing, Output: userID, roomID
const JoinMeeting = () => {
  const router = useRouter();
  const [formData, setFormData] = useState({
    meetingId: 100,
    name: "ducanh",
    rememberName: false,
    noAudio: false,
    noVideo: false,
  });
  const [isTyped, setIstyped] = useState(false);
  const handleChange = (e: any) => {
    setIstyped(e.target.value.length > 0);

    const { name, type, checked, value } = e.target;
    setFormData((prev) => ({
      ...prev,
      [name]: type === "checkbox" ? checked : value,
    }));
  };

  const handleSubmit = (e: any) => {
    router.push(
      `/meeting/enter-meeting?roomID=${formData.meetingId}&userName=${formData.name}`
    );
  };
  return (
    <div className="min-h-screen bg-gradient-to-br from-indigo-900 via-purple-900 to-pink-800 flex flex-col items-center justify-center px-8 py-6">
      {/* Title */}
      <h1 className="text-white text-2xl font-medium mb-8">Join meeting</h1>

      {/* Form */}
      <div className="space-y-4 max-w-xl bg-white/10 backdrop-blur-lg rounded-2xl p-8 shadow-2xl border border-white/20">
        {/* Meeting ID Input */}
        <div className="relative">
          <input
            type="text"
            name="meetingId"
            placeholder="Meeting ID or personal link name"
            value={formData.meetingId}
            onChange={handleChange}
            className="w-full px-4 py-3 bg-white/10 border border-white/20 rounded-xl focus:border-pink-500 focus:ring-1 focus:ring-pink-500/30 outline-none text-white placeholder-white/50"
          />
          <button className="absolute right-3 top-1/2 -translate-y-1/2 text-white/50 hover:text-white/80">
            <svg
              xmlns="http://www.w3.org/2000/svg"
              fill="none"
              viewBox="0 0 24 24"
              strokeWidth={1.5}
              stroke="currentColor"
              className="w-5 h-5"
            >
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                d="m19.5 8.25-7.5 7.5-7.5-7.5"
              />
            </svg>
          </button>
        </div>

        {/* Name Input */}
        <div>
          <input
            type="text"
            name="name"
            value={formData.name}
            onChange={handleChange}
            className="w-full px-4 py-3 bg-white/10 border border-white/20 rounded-xl focus:border-pink-500 focus:ring-1 focus:ring-pink-500/30 outline-none text-white placeholder-white/50"
          />
        </div>

        {/* Checkboxes */}
        <div className="space-y-3">
          {/* Remember name */}
          <label className="flex items-center gap-2 cursor-pointer group">
            <div className="relative">
              <input
                type="checkbox"
                name="rememberName"
                checked={formData.rememberName}
                onChange={handleChange}
                className="w-4 h-4 border border-white/20 rounded appearance-none checked:bg-gradient-to-r from-pink-500 to-purple-500 checked:border-transparent cursor-pointer"
              />
              {formData.rememberName && (
                <svg
                  className="absolute top-0 left-0 w-4 h-4 text-white pointer-events-none"
                  xmlns="http://www.w3.org/2000/svg"
                  viewBox="0 0 24 24"
                  fill="none"
                  stroke="currentColor"
                  strokeWidth="2"
                >
                  <polyline points="20 6 9 17 4 12"></polyline>
                </svg>
              )}
            </div>
            <span className="text-white/80 group-hover:text-white">
              Remember me
            </span>
          </label>

          {/* Don't connect audio */}
          <label className="flex items-center gap-2 cursor-pointer group">
            <div className="relative">
              <input
                type="checkbox"
                name="noAudio"
                checked={formData.noAudio}
                onChange={handleChange}
                className="w-4 h-4 border border-white/20 rounded appearance-none checked:bg-gradient-to-r from-pink-500 to-purple-500 checked:border-transparent cursor-pointer"
              />
              {formData.noAudio && (
                <svg
                  className="absolute top-0 left-0 w-4 h-4 text-white pointer-events-none"
                  xmlns="http://www.w3.org/2000/svg"
                  viewBox="0 0 24 24"
                  fill="none"
                  stroke="currentColor"
                  strokeWidth="2"
                >
                  <polyline points="20 6 9 17 4 12"></polyline>
                </svg>
              )}
            </div>
            <span className="text-white/80 group-hover:text-white">Audio</span>
          </label>

          {/* Turn off video */}
          <label className="flex items-center gap-2 cursor-pointer group">
            <div className="relative">
              <input
                type="checkbox"
                name="noVideo"
                checked={formData.noVideo}
                onChange={handleChange}
                className="w-4 h-4 border border-white/20 rounded appearance-none checked:bg-gradient-to-r from-pink-500 to-purple-500 checked:border-transparent cursor-pointer"
              />
              {formData.noVideo && (
                <svg
                  className="absolute top-0 left-0 w-4 h-4 text-white pointer-events-none"
                  xmlns="http://www.w3.org/2000/svg"
                  viewBox="0 0 24 24"
                  fill="none"
                  stroke="currentColor"
                  strokeWidth="2"
                >
                  <polyline points="20 6 9 17 4 12"></polyline>
                </svg>
              )}
            </div>
            <span className="text-white/80 group-hover:text-white">Camera</span>
          </label>
        </div>

        {/* Terms Text */}
        <p className="text-sm text-white/80">
          By clicking "Join", I accept with{" "}
          <a href="#" className="text-white hover:underline">
            Policy Statement
          </a>{" "}
          and{" "}
          <a href="#" className="text-white hover:underline">
            Private Security
          </a>
          .
        </p>

        {/* Action Buttons */}
        <div className="flex justify-end gap-3 pt-4">
          <button
            className="px-6 py-2 border border-white/20 rounded-xl text-white/80 hover:bg-white/10 transition-colors"
            onClick={() => router.back()}
          >
            Cancel
          </button>
          <button
            className={`px-6 py-2 rounded-xl ${
              isTyped
                ? "bg-gradient-to-r from-pink-500 to-purple-500 text-white hover:from-pink-600 hover:to-purple-600"
                : "bg-white/10 text-white/50"
            } transition-all duration-300`}
            onClick={handleSubmit}
            disabled={!formData.meetingId}
          >
            Join
          </button>
        </div>
      </div>
    </div>
  );
};

export default JoinMeeting;
