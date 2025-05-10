"use client";
import CalendarSideBar from "@/app/component/home/CalendarSideBar";
import TopNavigationBar from "@/app/component/navigation/TopNavigationBar";
import React from "react";
import { useState } from "react";

const ZoomHome = () => {
  const [onSchedule, setOnSchedule] = useState(false);
  const handleSchedule = () => {
    console.log("Handle Schedule");
    setOnSchedule(!onSchedule);
  };
  return (
    <div className="flex flex-col h-screen bg-white">
      {/* Top Navigation Bar */}
      <TopNavigationBar />

      {/* Existing content */}
      <div className="flex flex-1 justify-between">
        <iframe
          src="https://onlineboard.eu/b/RXjMlmtO"
          width="1000"
          height="600"
          frameBorder={0}
          scrolling="no"
        ></iframe>

        {/* Calendar sidebar */}
        <CalendarSideBar />
      </div>
    </div>
  );
};

export default ZoomHome;
