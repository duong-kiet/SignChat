import React from "react";

interface SignInButtonProps {
  isLoading: boolean;
  onSignIn: (e: React.MouseEvent) => void;
}

const SignInButton: React.FC<SignInButtonProps> = ({ isLoading, onSignIn }) => {
  return (
    <button
      onClick={onSignIn}
      disabled={isLoading}
      className="w-full py-3 bg-gradient-to-r from-pink-500 to-purple-500 text-white rounded-xl hover:from-pink-600 hover:to-purple-600 transition-all duration-300 font-medium"
    >
      {isLoading ? "Signing in..." : "Sign In"}
    </button>
  );
};

export default SignInButton;
