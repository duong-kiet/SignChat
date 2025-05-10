import TopNavigationBar from "@/app/component/navigation/TopNavigationBar";

const SignTranslatePage = () => {
  return (
    <div className="flex flex-col h-screen bg-white">
      {/* Top Navigation Bar */}
      <div className="relative z-10">
        <TopNavigationBar />
      </div>

      {/* Existing content */}
      <div className="flex flex-1 justify-between relative -mt-16">
        <iframe
          src="https://sign.mt"
          width="100%"
          height="100%"
          className="z-0"
        />
      </div>
    </div>
  );
};

export default SignTranslatePage;
