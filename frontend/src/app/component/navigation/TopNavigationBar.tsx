"use client";
import Link from "next/link";
import { usePathname } from "next/navigation";
import { handleSignOut } from "@/app/helpers/firebase/logout";
const TopNavigationBar = () => {
  const pathname = usePathname();

  return (
    <>
      <div className="flex items-center h-12 bg-[#F2F2F2] px-4 border-b">
        <div className="flex items-center gap-4">
          {/* Logo and Title */}
          <div className="flex items-center gap-2">
            <span className="font-bold text-gray-800">SignMeet</span>
            <span className="text-gray-600">Workplace</span>
          </div>

          {/* History button */}
          <button className="p-1 hover:bg-gray-200 rounded">
            <svg
              className="w-5 h-5 text-gray-600"
              fill="none"
              stroke="currentColor"
              viewBox="0 0 24 24"
            >
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                strokeWidth="2"
                d="M12 8v4l3 3m6-3a9 9 0 11-18 0 9 9 0 0118 0z"
              />
            </svg>
          </button>

          {/* Search */}
          <div className="flex items-center bg-white rounded px-2 py-1">
            <svg
              className="w-4 h-4 text-gray-500 mr-2"
              fill="none"
              stroke="currentColor"
              viewBox="0 0 24 24"
            >
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                strokeWidth="2"
                d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z"
              />
            </svg>
            <input
              type="text"
              placeholder="Search"
              className="outline-none text-sm w-32"
            />
            <span className="text-xs text-gray-500 ml-2">Ctrl+F</span>
          </div>
        </div>

        {/* Center Navigation */}
        <div className="flex-1 flex justify-center items-center">
          <div className="flex items-center bg-gray-200 rounded-lg p-1 gap-1">
            <Link
              href="/home"
              className={`flex items-center gap-2 px-4 py-1.5 rounded-md ${
                pathname === "/home" ? "bg-white" : "hover:bg-white"
              }`}
            >
              <svg
                className="w-5 h-5 text-gray-700"
                fill="none"
                stroke="currentColor"
                viewBox="0 0 24 24"
              >
                <path
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  strokeWidth="2"
                  d="M3 12l2-2m0 0l7-7 7 7M5 10v10a1 1 0 001 1h3m10-11l2 2m-2-2v10a1 1 0 01-1 1h-3m-6 0a1 1 0 001-1v-4a1 1 0 011-1h2a1 1 0 011 1v4a1 1 0 001 1m-6 0h6"
                />
              </svg>
              <span className="text-sm">Home</span>
            </Link>

            <Link
              href="/calendar"
              className={`flex items-center gap-2 px-4 py-1.5 rounded-md ${
                pathname === "/calendar" ? "bg-white" : "hover:bg-white"
              }`}
            >
              <svg
                className="w-5 h-5 text-gray-700"
                fill="none"
                stroke="currentColor"
                viewBox="0 0 24 24"
              >
                <path
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  strokeWidth="2"
                  d="M8 7V3m8 4V3m-9 8h10M5 21h14a2 2 0 002-2V7a2 2 0 00-2-2H5a2 2 0 00-2 2v12a2 2 0 002 2z"
                />
              </svg>
              <span className="text-sm">Calendar</span>
            </Link>

            <Link
              href="/sign-translate"
              className={`flex items-center gap-2 px-4 py-1.5 rounded-md ${
                pathname === "/team-chat" ? "bg-white" : "hover:bg-white"
              }`}
            >
              <svg
                className="w-5 h-5 text-gray-700"
                fill="none"
                stroke="currentColor"
                viewBox="0 0 24 24"
              >
                <path
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  strokeWidth="2"
                  d="M17 8h2a2 2 0 012 2v6a2 2 0 01-2 2h-2v4l-4-4H9a1.994 1.994 0 01-1.414-.586m0 0L11 14h4a2 2 0 002-2V6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2v4l.586-.586z"
                />
              </svg>
              <span className="text-sm">Sign Translate</span>
            </Link>

            <Link
              href="/whiteboards"
              className={`flex items-center gap-2 px-4 py-1.5 rounded-md ${
                pathname === "/whiteboards" ? "bg-white" : "hover:bg-white"
              }`}
            >
              <svg
                className="w-5 h-5 text-gray-700"
                fill="none"
                stroke="currentColor"
                viewBox="0 0 24 24"
              >
                <path
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  strokeWidth="2"
                  d="M15.232 5.232l3.536 3.536m-2.036-5.036a2.5 2.5 0 113.536 3.536L6.5 21.036H3v-3.572L16.732 3.732z"
                />
              </svg>
              <span className="text-sm">Whiteboards</span>
            </Link>

            <Link
              href="/docs"
              className={`flex items-center gap-2 px-4 py-1.5 rounded-md ${
                pathname === "/docs" ? "bg-white" : "hover:bg-white"
              }`}
            >
              <svg
                className="w-5 h-5 text-gray-700"
                fill="none"
                stroke="currentColor"
                viewBox="0 0 24 24"
              >
                <path
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  strokeWidth="2"
                  d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z"
                />
              </svg>
              <span className="text-sm">Docs</span>
            </Link>
          </div>
        </div>

        {/* Right side icons */}
        <div className="flex items-center gap-2">
          <button
            onClick={(e) => {
              handleSignOut();
            }}
            className="p-2 hover:bg-gray-200 rounded"
          >
            <svg
              className="w-5 h-5 text-gray-600"
              fill="none"
              stroke="currentColor"
              viewBox="0 0 24 24"
            >
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                strokeWidth="2"
                d="M17 16l4-4m0 0l-4-4m4 4H7m6 4v1a3 3 0 01-3 3H6a3 3 0 01-3-3V7a3 3 0 013-3h4a3 3 0 013 3v1"
              />
            </svg>
          </button>
          <button className="p-2 hover:bg-gray-200 rounded">
            <div className="relative">
              <svg
                className="w-5 h-5 text-gray-600"
                fill="none"
                stroke="currentColor"
                viewBox="0 0 24 24"
              >
                <path
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  strokeWidth="2"
                  d="M15 17h5l-1.405-1.405A2.032 2.032 0 0118 14.158V11a6.002 6.002 0 00-4-5.659V5a2 2 0 10-4 0v.341C7.67 6.165 6 8.388 6 11v3.159c0 .538-.214 1.055-.595 1.436L4 17h5m6 0v1a3 3 0 11-6 0v-1m6 0H9"
                />
              </svg>
              <span className="absolute -top-1 -right-1 bg-red-500 text-white text-xs rounded-full w-4 h-4 flex items-center justify-center">
                1
              </span>
            </div>
          </button>
        </div>
      </div>
    </>
  );
};
export default TopNavigationBar;
