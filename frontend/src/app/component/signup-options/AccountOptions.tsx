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
            className="flex flex-col items-center justify-center p-4 border border-gray-200 rounded-lg hover:bg-gray-50"
          >
            <div className="w-6 h-6 mb-2">
              {provider === "SSO" && (
                <svg
                  xmlns="http://www.w3.org/2000/svg"
                  fill="none"
                  viewBox="0 0 24 24"
                  strokeWidth={1.5}
                  stroke="currentColor"
                >
                  <path
                    strokeLinecap="round"
                    strokeLinejoin="round"
                    d="m21 21-5.197-5.197m0 0A7.5 7.5 0 1 0 5.196 5.196a7.5 7.5 0 0 0 10.607 10.607Z"
                  />
                </svg>
              )}
              {/* Add other provider icons here */}
              {provider === "Apple" && <FaApple className="w-6 h-6" />}
              {provider === "Google" && <FcGoogle className="w-6 h-6" />}
              {provider === "Facebook" && (
                <FaFacebook className="w-6 h-6 text-blue-700" />
              )}
            </div>
            <span className="text-xs text-gray-600">{provider}</span>
          </button>
        ))}
      </div>
    </>
  );
};
export default AccountOptions;
