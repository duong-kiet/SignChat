// components/RemoteVideoComponent.js
import React from "react";

const RemoteVideoComponent = React.forwardRef((props, ref) => (
  <div className="remote-video-container w-full lg:w-[25%] aspect-video min-w-[200px] flex items-center justify-center">
    <video
      id="remote-video"
      ref={ref}
      className="w-full h-full object-cover rounded-lg"
      autoPlay
      playsInline
    ></video>
  </div>
));

export default RemoteVideoComponent;
