const LocalVideoComponent = () => (
  <div className="local-video-container w-full lg:w-[70%] aspect-video flex items-center">
    <video
      id="local-video"
      className="w-full h-full object-cover rounded-lg"
      autoPlay
      playsInline
      muted
    ></video>
  </div>
);

export default LocalVideoComponent;
