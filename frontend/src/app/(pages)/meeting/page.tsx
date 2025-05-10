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
    <div className="min-h-screen bg-white flex flex-col items-center justify-center px-8 py-6">
      {/* Title */}
      <h1 className="text-[#232333] text-2xl font-medium mb-8">Join meeting</h1>

      {/* Form */}
      <div className="space-y-4 max-w-xl">
        {/* Meeting ID Input */}
        <div className="relative">
          <input
            type="text"
            name="meetingId"
            placeholder="Meeting ID or personal link name"
            value={formData.meetingId}
            onChange={handleChange}
            className="w-full px-4 py-3 border border-[#0b5cff] rounded-lg focus:outline-none text-gray-600"
          />
          <button className="absolute right-3 top-1/2 -translate-y-1/2 text-gray-400">
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
            className="w-full px-4 py-3 border border-gray-300 rounded-lg focus:border-[#0b5cff] focus:outline-none text-gray-600"
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
                className="w-4 h-4 border border-gray-300 rounded appearance-none checked:bg-[#0b5cff] checked:border-[#0b5cff] cursor-pointer"
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
            <span className="text-gray-600 group-hover:text-gray-800">
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
                className="w-4 h-4 border border-gray-300 rounded appearance-none checked:bg-[#0b5cff] checked:border-[#0b5cff] cursor-pointer"
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
            <span className="text-gray-600 group-hover:text-gray-800">
              Audio
            </span>
          </label>

          {/* Turn off video */}
          <label className="flex items-center gap-2 cursor-pointer group">
            <div className="relative">
              <input
                type="checkbox"
                name="noVideo"
                checked={formData.noVideo}
                onChange={handleChange}
                className="w-4 h-4 border border-gray-300 rounded appearance-none checked:bg-[#0b5cff] checked:border-[#0b5cff] cursor-pointer"
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
            <span className="text-gray-600 group-hover:text-gray-800">
              Camera
            </span>
          </label>
        </div>

        {/* Terms Text */}
        <p className="text-sm text-gray-600">
          By clicking "Join", I accept with{" "}
          <a href="#" className="text-[#0b5cff] hover:underline">
            Policy Statement
          </a>{" "}
          and{" "}
          <a href="#" className="text-[#0b5cff] hover:underline">
            Private Security
          </a>
          .
        </p>

        {/* Action Buttons */}
        <div className="flex justify-end gap-3 pt-4">
          <button
            className="px-6 py-2 border border-gray-300 rounded-lg text-gray-600 hover:bg-gray-50"
            onClick={() => router.back()}
          >
            Cancel
          </button>
          <button
            className={`px-6 py-2  rounded-lg ${
              isTyped ? "bg-blue-700 text-white" : " bg-[#f3f3f4] text-gray-400"
            }`}
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
