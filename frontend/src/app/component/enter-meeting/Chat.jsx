"use client";
import { useState, useEffect } from "react";
import { auth } from "@/app/helpers/firebase/firebase";
import {
  sendMessage,
  clearChat,
  subscribeToMessages,
} from "@/app/helpers/firebase/chat";

const Chat = ({ isChatOpen }) => {
  const [messages, setMessages] = useState([]);
  const [isChat, setIsChat] = useState(isChatOpen);

  const handleSendMessage = async (e) => {
    e.preventDefault();
    const { content, userName, userId } = await sendMessage(currentMessage);

    setMessages((prevMessages) => [
      ...prevMessages,
      { text: content, sender: userName, userId: userId },
    ]);
    setCurrentMessage("");
  };

  const handleCloseChat = () => {
    setIsChat(!isChat);
    clearChat();
  };

  useEffect(() => {
    const unsubscribe = subscribeToMessages((message) => {
      setMessages((prevMessages) => [...prevMessages, message]);
    });

    return () => unsubscribe();
  }, []);

  const [currentMessage, setCurrentMessage] = useState("");
  return (
    <>
      {isChat && (
        <form className="fixed top-0 right-0 h-screen w-80 bg-white flex flex-col shadow-lg z-50">
          <div className="p-3 border-b flex justify-between items-center bg-gray-50">
            <h2 className="text-base font-medium text-gray-700">
              Meeting Chat
            </h2>
            <button
              className="p-1 hover:bg-gray-200 rounded-full transition duration-200"
              onClick={() => handleCloseChat()}
            >
              <svg
                className="w-5 h-5 text-gray-500"
                fill="none"
                stroke="currentColor"
                viewBox="0 0 24 24"
              >
                <path
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  strokeWidth={2}
                  d="M6 18L18 6M6 6l12 12"
                />
              </svg>
            </button>
          </div>

          <div className="flex-1 overflow-y-auto px-4 py-3">
            {messages.map((msg, index) => (
              <div
                key={index}
                className={`mb-4 flex ${
                  msg.userId === auth.currentUser?.uid
                    ? "justify-end"
                    : "justify-start"
                }`}
              >
                <div
                  className={`max-w-[85%] ${
                    msg.userId === auth.currentUser?.uid
                      ? "text-right"
                      : "text-left"
                  }`}
                >
                  <div className="text-xs text-gray-500 mb-1">{msg.sender}</div>
                  <div
                    className={`rounded-lg px-3 py-2 inline-block ${
                      msg.userId === auth.currentUser?.uid
                        ? "bg-blue-500 text-white"
                        : "bg-gray-100 text-gray-800"
                    }`}
                  >
                    <p className="text-sm">{msg.text}</p>
                  </div>
                </div>
              </div>
            ))}
          </div>

          <div className="p-4 border-t bg-gray-50">
            <div className="flex flex-col space-y-3">
              <div className="flex items-center space-x-2">
                <span className="text-sm text-gray-600 font-medium">To:</span>
                <select className="text-sm bg-transparent border-0 focus:ring-0 text-gray-600 font-medium">
                  <option>Everyone</option>
                </select>
              </div>

              <div className="relative">
                <input
                  type="text"
                  className="w-full border rounded-lg px-4 py-2.5 pr-12 focus:outline-none focus:border-blue-500 focus:ring-1 focus:ring-blue-500 text-sm text-black"
                  placeholder="Type message here..."
                  value={currentMessage}
                  onChange={(e) => setCurrentMessage(e.target.value)}
                  onKeyDown={(e) => e.key === "Enter" && handleSendMessage(e)}
                />
                <button
                  onClick={handleSendMessage}
                  className="absolute right-2 top-1/2 -translate-y-1/2 text-blue-500 hover:text-blue-600"
                >
                  <svg
                    className="w-5 h-5"
                    fill="none"
                    stroke="currentColor"
                    viewBox="0 0 24 24"
                  >
                    <path
                      strokeLinecap="round"
                      strokeLinejoin="round"
                      strokeWidth={2}
                      d="M12 19l9 2-9-18-9 18 9-2zm0 0v-8"
                    />
                  </svg>
                </button>
              </div>

              <div className="flex items-center space-x-2 px-1">
                <button className="p-1.5 hover:bg-gray-200 rounded-full transition-colors">
                  <svg
                    className="w-5 h-5 text-gray-600"
                    fill="none"
                    stroke="currentColor"
                    viewBox="0 0 24 24"
                  >
                    <path
                      strokeLinecap="round"
                      strokeLinejoin="round"
                      strokeWidth={2}
                      d="M15.172 7l-6.586 6.586a2 2 0 102.828 2.828l6.414-6.586a4 4 0 00-5.656-5.656l-6.415 6.585a6 6 0 108.486 8.486L20.5 13"
                    />
                  </svg>
                </button>
                <button className="p-1.5 hover:bg-gray-200 rounded-full transition-colors">
                  <svg
                    className="w-5 h-5 text-gray-600"
                    fill="none"
                    stroke="currentColor"
                    viewBox="0 0 24 24"
                  >
                    <path
                      strokeLinecap="round"
                      strokeLinejoin="round"
                      strokeWidth={2}
                      d="M14.828 14.828a4 4 0 01-5.656 0M9 10h.01M15 10h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z"
                    />
                  </svg>
                </button>
                <button className="p-1.5 hover:bg-gray-200 rounded-full transition-colors">
                  <svg
                    className="w-5 h-5 text-gray-600"
                    fill="none"
                    stroke="currentColor"
                    viewBox="0 0 24 24"
                  >
                    <path
                      strokeLinecap="round"
                      strokeLinejoin="round"
                      strokeWidth={2}
                      d="M4 6h16M4 12h16m-7 6h7"
                    />
                  </svg>
                </button>
              </div>
            </div>
          </div>
        </form>
      )}
    </>
  );
};

export default Chat;
