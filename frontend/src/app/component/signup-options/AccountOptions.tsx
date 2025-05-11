import { FaApple } from "react-icons/fa";
import { FcGoogle } from "react-icons/fc";
import { FaFacebook } from "react-icons/fa";

const AccountOptions = () => {
  return (
    <>
      {/* Social Sign Up Options */}
      <div className="grid grid-cols-4 gap-4">
        {["SSO", "Apple", "Google", "Facebook"].map((provider) => (
          <button
            key={provider}
            className="group flex flex-col items-center justify-center p-4 border border-white/20 rounded-lg bg-white/10 hover:bg-white transition-all duration-300"
          >
            <div className="w-6 h-6 mb-2">
              {provider === "SSO" && (
                <svg
                  xmlns="http://www.w3.org/2000/svg"
                  fill="none"
                  viewBox="0 0 24 24"
                  strokeWidth={1.5}
                  stroke="currentColor"
                  className="text-white group-hover:text-black"
                >
                  <path
                    strokeLinecap="round"
                    strokeLinejoin="round"
                    d="m21 21-5.197-5.197m0 0A7.5 7.5 0 1 0 5.196 5.196a7.5 7.5 0 0 0 10.607 10.607Z"
                  />
                </svg>
              )}
              {/* Add other provider icons here */}
              {provider === "Apple" && (
                <FaApple className="w-6 h-6 text-white group-hover:text-black" />
              )}
              {provider === "Google" && <FcGoogle className="w-6 h-6" />}
              {provider === "Facebook" && (
                <FaFacebook className="w-6 h-6 text-[#1877F2]" />
              )}
            </div>
            <span className="text-xs text-white group-hover:text-black">
              {provider}
            </span>
          </button>
        ))}
      </div>
    </>
  );
};

export default AccountOptions;
