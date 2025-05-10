"use client";

import React, { useState, useEffect } from "react";
import ToolBar from "@/app/component/enter-meeting/ToolBar";
import VideoArea from "@/app/component/enter-meeting/VideoArea";
import { createZegoEngine } from "@/app/helpers/zegoEngineManager";
import { ZEGO_CONFIG } from "@/app/config/zego.config";
import { ZegoExpressEngine } from "zego-express-engine-webrtc";

const VideoCallComponent = () => {
  const [isSubtitle, setIsSubtitle] = useState(false);
  const [isSignLanguage, setIsSignLanguage] = useState(false);
  const [selectedLanguage, setSelectedLanguage] = useState("off");
  const [zegoEngine, setZegoEngine] = useState<ZegoExpressEngine | null>(null);
  const [localStream, setLocalStream] = useState<any>(null);

  useEffect(() => {
    try {
      const zg = createZegoEngine(
        ZEGO_CONFIG.appID,
        ZEGO_CONFIG.server,
        ZEGO_CONFIG.logConfig
      );
      setZegoEngine(zg);

      // Cleanup function
      return () => {
        if (zg && localStream) {
          zg.destroyStream(localStream);
          zg.logoutRoom();
        }
      };
    } catch (error) {
      console.error("Failed to initialize Zego Engine:", error);
    }
  }, []);

  return (
    <div className="flex flex-col bg-gray-900 text-white h-screen">
      <div className="flex-1">
        <VideoArea
          isSubtitle={isSubtitle}
          isSignLanguage={isSignLanguage}
          selectedLanguage={selectedLanguage}
          zegoEngine={zegoEngine}
        />
      </div>
      <ToolBar
        isSubtitle={isSubtitle}
        setIsSubtitle={setIsSubtitle}
        isSignLanguage={isSignLanguage}
        setIsSignLanguage={setIsSignLanguage}
        selectedLanguage={selectedLanguage}
        setSelectedLanguage={setSelectedLanguage}
      />
    </div>
  );
};

export default VideoCallComponent;
