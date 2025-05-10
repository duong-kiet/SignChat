"use client";

import AccountOptions from "@/app/component/signup-options/AccountOptions";
import Link from "next/link";
import React, { useState } from "react";
import { handleSignUp } from "@/app/helpers/firebase/signup";
const ZoomSignUp = () => {
  const [showPassword, setShowPassword] = useState(false);
  const [agreeTerms, setAgreeTerms] = useState(false);
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [firstName, setFirstName] = useState("");
  const [lastName, setLastName] = useState("");
  const [username, setUsername] = useState("");
  const [isLoading, setIsLoading] = useState(false);

  const handleNavigation = () => {
    setIsLoading(true);
  };

  return (
    <div className="min-h-screen flex flex-col items-center py-8 px-4">
      {/* Header */}
      <div className="text-center space-y-1 mb-12">
        <div className="flex items-center justify-center gap-2">
          <h1 className="text-[#0b5cff] text-3xl font-bold">SignMeet</h1>
          <button className="text-gray-500">
            <svg
              xmlns="http://www.w3.org/2000/svg"
              fill="none"
              viewBox="0 0 24 24"
              strokeWidth={1.5}
              stroke="currentColor"
              className="w-5 h-5"
            >
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                d="m19.5 8.25-7.5 7.5-7.5-7.5"
              />
            </svg>
          </button>
        </div>
        <h2 className="text-[#232333] text-3xl">Workplace</h2>
        <p className="text-gray-500 text-sm">us05web.zoom.us</p>
      </div>

      {/* Form Container */}
      <form className="w-full max-w-md space-y-4">
        {/* Name Inputs */}
        <div className="grid grid-cols-2 gap-4">
          <div className="relative">
            <input
              type="text"
              placeholder="First Name"
              className="w-full px-4 py-3 border border-gray-300 rounded-lg focus:border-blue-500 focus:ring-1 focus:ring-blue-500 outline-none"
              value={firstName}
              onChange={(e) => setFirstName(e.target.value)}
            />
          </div>
          <div className="relative">
            <input
              type="text"
              placeholder="Last Name"
              className="w-full px-4 py-3 border border-gray-300 rounded-lg focus:border-blue-500 focus:ring-1 focus:ring-blue-500 outline-none"
              value={lastName}
              onChange={(e) => setLastName(e.target.value)}
            />
          </div>
        </div>

        {/* Email Input */}
        <div className="relative">
          <input
            type="email"
            placeholder="Work Email"
            className="w-full px-4 py-3 border border-gray-300 rounded-lg focus:border-blue-500 focus:ring-1 focus:ring-blue-500 outline-none"
            value={email}
            onChange={(e) => setEmail(e.target.value)}
          />
        </div>
        {/* Username Input */}
        <div className="relative">
          <input
            type="text"
            placeholder="Username"
            className="w-full px-4 py-3 border border-gray-300 rounded-lg focus:border-blue-500 focus:ring-1 focus:ring-blue-500 outline-none"
            value={username}
            onChange={(e) => setUsername(e.target.value)}
          />
        </div>

        {/* Password Input */}
        <div className="relative">
          <input
            type={showPassword ? "text" : "password"}
            placeholder="Password (8 characters min.)"
            className="w-full px-4 py-3 border border-gray-300 rounded-lg focus:border-blue-500 focus:ring-1 focus:ring-blue-500 outline-none pr-12"
            value={password}
            onChange={(e) => setPassword(e.target.value)}
          />
          <button
            onClick={() => setShowPassword(!showPassword)}
            className="absolute right-3 top-1/2 -translate-y-1/2 text-gray-500 hover:text-gray-700"
          >
            {showPassword ? (
              <svg
                xmlns="http://www.w3.org/2000/svg"
                fill="none"
                viewBox="0 0 24 24"
                strokeWidth={1.5}
                stroke="currentColor"
                className="w-5 h-5"
              >
                <path
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  d="M3.98 8.223A10.477 10.477 0 001.934 12C3.226 16.338 7.244 19.5 12 19.5c.993 0 1.953-.138 2.863-.395M6.228 6.228A10.45 10.45 0 0112 4.5c4.756 0 8.773 3.162 10.065 7.498a10.523 10.523 0 01-4.293 5.774M6.228 6.228L3 3m3.228 3.228l3.65 3.65m7.894 7.894L21 21m-3.228-3.228l-3.65-3.65m0 0a3 3 0 10-4.243-4.243m4.242 4.242L9.88 9.88"
                />
              </svg>
            ) : (
              <svg
                xmlns="http://www.w3.org/2000/svg"
                fill="none"
                viewBox="0 0 24 24"
                strokeWidth={1.5}
                stroke="currentColor"
                className="w-5 h-5"
              >
                <path
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  d="M2.036 12.322a1.012 1.012 0 010-.639C3.423 7.51 7.36 4.5 12 4.5c4.638 0 8.573 3.007 9.963 7.178.07.207.07.431 0 .639C20.577 16.49 16.64 19.5 12 19.5c-4.638 0-8.573-3.007-9.963-7.178z"
                />
                <path
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  d="M15 12a3 3 0 11-6 0 3 3 0 016 0z"
                />
              </svg>
            )}
          </button>
        </div>

        {/* Terms Checkbox */}
        <div className="flex items-start gap-2">
          <input
            type="checkbox"
            id="terms"
            checked={agreeTerms}
            onChange={(e) => setAgreeTerms(e.target.checked)}
            className="mt-1 w-4 h-4 rounded border-gray-300"
          />
          <label
            htmlFor="terms"
            className="text-sm text-gray-600 leading-tight"
          >
            By signing up, I agree to the{" "}
            <a href="#" className="text-blue-500 hover:underline">
              Terms of Service
            </a>{" "}
            and{" "}
            <a href="#" className="text-blue-500 hover:underline">
              Privacy Statement
            </a>
            .
          </label>
        </div>

        {/* Sign Up Button */}
        <button
          className="w-full py-3 bg-[#2D8CFF] text-white rounded-lg hover:bg-blue-600 transition-colors"
          onClick={async (e) => {
            setIsLoading(true);
            const success = await handleSignUp(
              e,
              email,
              password,
              firstName,
              lastName,
              username
            );
            if (success) {
              window.location.href = "/home";
            }
            setIsLoading(false);
          }}
          disabled={isLoading}
        >
          {isLoading ? "Creating account..." : "Sign Up"}
        </button>

        {/* Or sign up with */}
        <div className="relative text-center my-8">
          <div className="absolute inset-0 flex items-center">
            <div className="w-full border-t border-gray-200"></div>
          </div>
          <span className="relative bg-white px-4 text-sm text-gray-500">
            or sign up with
          </span>
        </div>

        {/* Social Sign Up Options */}
        <AccountOptions />
      </form>

      {/* Footer */}
      <div className="mt-auto w-full flex justify-between items-center px-4">
        <div className="w-full bottom-0 left-0 right-0 p-4 flex justify-between bg-white">
          <button
            className="text-gray-600 hover:text-gray-900 flex items-center"
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
          <Link href="/signin">
            <button
              className="text-blue-500 hover:text-blue-600"
              disabled={isLoading}
            >
              Sign In
            </button>
          </Link>
        </div>
      </div>
    </div>
  );
};

export default ZoomSignUp;
