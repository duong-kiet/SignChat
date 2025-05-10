"use client";
import CalendarSideBar from "@/app/component/home/CalendarSideBar";
import ScheduleMeetingForm from "@/app/component/home/ScheduleMeetingForm";
import TopNavigationBar from "@/app/component/navigation/TopNavigationBar";
import Link from "next/link";
import React from "react";
import { useState } from "react";

const ZoomHome = () => {
  const [onSchedule, setOnSchedule] = useState(false);
  const [isLoading, setIsLoading] = useState(false);

  const handleSchedule = () => {
    console.log("Handle Schedule");
    setOnSchedule(!onSchedule);
  };
  return (
    <div className="flex flex-col h-screen bg-white">
      {/* Top Navigation Bar */}
      <TopNavigationBar />

      {/* Existing content */}
      <div className="flex flex-1">
        {/* Main content */}
        {onSchedule && (
          <div className="fixed inset-0 flex items-center justify-center bg-black bg-opacity-50 z-50">
            <div className="relative bg-white p-8 rounded-lg shadow-lg max-h-full overflow-y-auto">
              {/* Nút "X" để đóng */}
              <button
                onClick={handleSchedule}
                className="absolute top-1 right-4 text-[48px] text-gray-500 hover:text-gray-700"
              >
                &times;
              </button>

              <ScheduleMeetingForm />
            </div>
          </div>
        )}

        <div className="flex-1 flex items-center justify-center p-8">
          {/* Quick actions grid */}
          <div className="grid grid-cols-2 gap-x-16 gap-y-8 max-w-xl mx-auto">
            {/* New meeting */}
            <Link
              href="/meeting/enter-meeting"
              onClick={(e) => {
                e.preventDefault();
                setIsLoading(true);
                setTimeout(() => {
                  window.location.href = "/meeting/enter-meeting";
                }, 500);
              }}
            >
              <div className="flex flex-col items-center">
                <button className="w-28 h-28 bg-[#F26D21] rounded-2xl flex items-center justify-center hover:bg-[#E65D15] transition-colors">
                  {isLoading ? (
                    <svg
                      className="w-14 h-14 text-white animate-spin"
                      viewBox="0 0 24 24"
                    >
                      <circle
                        className="opacity-25"
                        cx="12"
                        cy="12"
                        r="10"
                        stroke="currentColor"
                        strokeWidth="4"
                        fill="none"
                      />
                      <path
                        className="opacity-75"
                        fill="currentColor"
                        d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"
                      />
                    </svg>
                  ) : (
                    <svg
                      className="w-14 h-14 text-white"
                      fill="none"
                      stroke="currentColor"
                      viewBox="0 0 24 24"
                    >
                      <path
                        strokeLinecap="round"
                        strokeLinejoin="round"
                        strokeWidth="2"
                        d="M15 10l4.553-2.276A1 1 0 0121 8.618v6.764a1 1 0 01-1.447.894L15 14M5 18h8a2 2 0 002-2V8a2 2 0 00-2-2H5a2 2 0 00-2 2v8a2 2 0 002 2z"
                      />
                    </svg>
                  )}
                </button>
                <div className="mt-3 text-[15px] text-gray-700 flex items-center">
                  <span>New meeting</span>
                  <svg
                    className="w-4 h-4 ml-1 mt-1"
                    fill="none"
                    stroke="currentColor"
                    viewBox="0 0 24 24"
                  >
                    <path
                      strokeLinecap="round"
                      strokeLinejoin="round"
                      strokeWidth="2"
                      d="M19 9l-7 7-7-7"
                    />
                  </svg>
                </div>
              </div>
            </Link>

            {/* Join */}
            <Link href={"/meeting"}>
              <div className="flex flex-col items-center">
                <button className="w-28 h-28 bg-blue-500 rounded-2xl flex items-center justify-center hover:bg-[#0747A6] transition-colors">
                  <svg
                    className="w-14 h-14 text-white"
                    fill="none"
                    stroke="currentColor"
                    viewBox="0 0 24 24"
                  >
                    <path
                      strokeLinecap="round"
                      strokeLinejoin="round"
                      strokeWidth="2"
                      d="M12 4v16m8-8H4"
                    />
                  </svg>
                </button>
                <span className="mt-3 text-[15px] text-gray-700">Join</span>
              </div>
            </Link>

            {/* Schedule */}
            <div className="flex flex-col items-center">
              <button
                onClick={handleSchedule}
                className="w-28 h-28 bg-blue-500 rounded-2xl flex items-center justify-center hover:bg-[#0747A6] transition-colors"
              >
                <svg
                  className="w-14 h-14 text-white"
                  fill="none"
                  stroke="currentColor"
                  viewBox="0 0 24 24"
                >
                  <path
                    strokeLinecap="round"
                    strokeLinejoin="round"
                    strokeWidth="2"
                    d="M8 7V3m8 4V3m-9 8h10M5 21h14a2 2 0 002-2V7a2 2 0 00-2-2H5a2 2 0 00-2 2v12a2 2 0 002 2z"
                  />
                </svg>
              </button>
              <span className="mt-3 text-[15px] text-gray-700">Schedule</span>
            </div>

            {/* Share screen */}
            <div className="flex flex-col items-center">
              <button className="w-28 h-28 bg-blue-500 rounded-2xl flex items-center justify-center hover:bg-[#0747A6] transition-colors">
                <svg
                  className="w-14 h-14 text-white"
                  fill="none"
                  stroke="currentColor"
                  viewBox="0 0 24 24"
                >
                  <path
                    strokeLinecap="round"
                    strokeLinejoin="round"
                    strokeWidth="2"
                    d="M8 7l4-4m0 0l4 4m-4-4v18"
                  />
                </svg>
              </button>
              <span className="mt-3 text-[15px] text-gray-700">
                Share screen
              </span>
            </div>
          </div>
        </div>

        {/* Calendar sidebar */}
        <CalendarSideBar />
      </div>
    </div>
  );
};

export default ZoomHome;
