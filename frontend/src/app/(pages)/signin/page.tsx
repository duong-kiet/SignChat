"use client";
import AccountOptions from "@/app/component/signup-options/AccountOptions";
import Link from "next/link";
import React, { useState } from "react";
import { handleSignIn } from "@/app/helpers/firebase/signin";
import EmailInput from "@/app/component/signin/EmailInput";
import PasswordInput from "@/app/component/signin/PasswordInput";
import SignInButton from "@/app/component/signin/SignInButton";

const ZoomSignIn = () => {
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [isLoading, setIsLoading] = useState(false);

  const handleNavigation = () => {
    setIsLoading(true);
  };

  const handleSignInClick = async (e: React.MouseEvent) => {
    setIsLoading(true);
    const success = await handleSignIn(e, email, password);
    if (success) {
      window.location.href = "/home";
    }
    setIsLoading(false);
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-indigo-900 via-purple-900 to-pink-800 flex flex-col items-center pt-16 px-4">
      {/* Logo */}
      <div className="mb-8 text-center">
        <div className="flex items-center justify-center">
          <h1 className="text-white text-3xl font-bold">SignMeet</h1>
          <svg
            className="w-4 h-4 ml-1 text-white/50"
            viewBox="0 0 24 24"
            fill="none"
            stroke="currentColor"
          >
            <path
              strokeLinecap="round"
              strokeLinejoin="round"
              strokeWidth="2"
              d="M19 9l-7 7-7-7"
            />
          </svg>
        </div>
        <h2 className="text-white text-2xl font-bold mt-2">Workplace</h2>
        <p className="text-white/60 text-sm mt-1">us05web.sigmeet.us</p>
      </div>

      {/* Sign in form */}
      <div className="w-full max-w-md">
        <form className="space-y-4 bg-white/10 backdrop-blur-lg rounded-2xl p-8 shadow-2xl border border-white/20">
          <EmailInput email={email} setEmail={setEmail} />
          <PasswordInput password={password} setPassword={setPassword} />
          <SignInButton isLoading={isLoading} onSignIn={handleSignInClick} />

          {/* Keep me signed in */}
          <div className="flex items-center">
            <input
              type="checkbox"
              id="keep-signed"
              className="w-4 h-4 border-white/20 rounded text-pink-500 focus:ring-pink-500/30"
            />
            <label htmlFor="keep-signed" className="ml-2 text-white/80">
              Remember me
            </label>
          </div>
        </form>

        {/* Social sign in */}
        <div className="mt-8">
          <div className="relative mb-6">
            <div className="absolute inset-0 flex items-center">
              <div className="w-full border-t border-white/20"></div>
            </div>
            <div className="relative flex justify-center text-sm">
              <span className="px-4 bg-gradient-to-br from-indigo-900 via-purple-900 to-pink-800 text-white/80">
                or sign in with
              </span>
            </div>
          </div>

          <AccountOptions />
        </div>
      </div>

      {/* Bottom navigation */}
      <div className="w-full bottom-0 left-0 right-0 p-4 flex justify-between bg-transparent">
        <button
          className="text-white/80 hover:text-white flex items-center"
          onClick={handleNavigation}
          disabled={isLoading}
        >
          <svg
            className="w-4 h-4 mr-1"
            fill="none"
            stroke="currentColor"
            viewBox="0 0 24 24"
          >
            <path
              strokeLinecap="round"
              strokeLinejoin="round"
              strokeWidth="2"
              d="M15 19l-7-7 7-7"
            />
          </svg>
          <Link href={"/"}>{isLoading ? "Loading..." : "Back"}</Link>
        </button>
        <Link href="/signup">
          <button className="text-white hover:text-white/80">Sign Up</button>
        </Link>
      </div>
    </div>
  );
};

export default ZoomSignIn;
