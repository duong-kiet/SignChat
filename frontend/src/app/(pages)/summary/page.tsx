"use client";
import React, { Fragment, useEffect } from "react";
import { useRouter } from "next/navigation";
import "./MeetingSummary.css"; // Import the CSS file
import Skeleton from "@mui/material/Skeleton";
import Image from "next/image";
import {
  FiHome,
  FiShare2,
  FiDownload,
  FiEdit,
  FiTrash,
  FiChevronDown,
} from "react-icons/fi";
import { Menu, Transition } from "@headlessui/react";

const languages = [
  { code: "vi", name: "Tiếng Việt", enabled: true },
  { code: "en", name: "English", enabled: true },
  { code: "ja", name: "日本語", enabled: true },
  { code: "ko", name: "한국어", enabled: true },
  { code: "zh", name: "中文", enabled: true },
  { code: "fr", name: "Français", enabled: true },
  { code: "de", name: "Deutsch", enabled: true },
  { code: "es", name: "Español", enabled: true },
  { code: "it", name: "Italiano", enabled: true },
  { code: "ru", name: "Русский", enabled: true },
  { code: "ar", name: "العربية", enabled: true },
  { code: "hi", name: "हिन्दी", enabled: true },
  { code: "pt", name: "Português", enabled: true },
  { code: "th", name: "ไทย", enabled: true },
  { code: "id", name: "Bahasa Indonesia", enabled: true },
];

// Thay thế nút chuyển đổi ngôn ngữ cũ bằng Menu dropdown mới
const LanguageSelector = ({
  language,
  setLanguage,
}: {
  language: string;
  setLanguage: (language: "vi" | "en") => void;
}) => {
  return (
    <Menu as="div" className="relative inline-block text-left">
      <div>
        <Menu.Button className="inline-flex w-full justify-center gap-x-1.5 rounded-md bg-white px-3 py-2 text-sm font-semibold text-gray-900 shadow-sm ring-1 ring-inset ring-gray-300 hover:bg-gray-50">
          {languages.find((lang) => lang.code === language)?.name}
          <FiChevronDown
            className="-mr-1 h-5 w-5 text-gray-400"
            aria-hidden="true"
          />
        </Menu.Button>
      </div>

      <Transition
        as={Fragment}
        enter="transition ease-out duration-100"
        enterFrom="transform opacity-0 scale-95"
        enterTo="transform opacity-100 scale-100"
        leave="transition ease-in duration-75"
        leaveFrom="transform opacity-100 scale-100"
        leaveTo="transform opacity-0 scale-95"
      >
        <Menu.Items className="absolute right-0 z-10 mt-2 w-40 origin-top-right rounded-md bg-white shadow-lg ring-1 ring-black ring-opacity-5 focus:outline-none">
          <div className="py-1">
            {languages.map((lang) => (
              <Menu.Item key={lang.code}>
                {({ active }) => (
                  <button
                    onClick={() =>
                      lang.enabled && setLanguage(lang.code as "vi" | "en")
                    }
                    className={`
                      ${active ? "bg-gray-100" : ""} 
                      ${!lang.enabled ? "opacity-50 cursor-not-allowed" : ""}
                      ${
                        language === lang.code
                          ? "bg-blue-50 text-blue-600"
                          : "text-gray-700"
                      }
                      group flex w-full items-center px-4 py-2 text-sm
                    `}
                    disabled={!lang.enabled}
                  >
                    {lang.name}
                    {!lang.enabled && (
                      <span className="ml-2 text-xs text-gray-400">
                        (Coming soon)
                      </span>
                    )}
                  </button>
                )}
              </Menu.Item>
            ))}
          </div>
        </Menu.Items>
      </Transition>
    </Menu>
  );
};

const MeetingSummary = () => {
  const router = useRouter();
  const [loading, setLoading] = React.useState(true);
  const [language, setLanguage] = React.useState<"vi" | "en">("vi");
  const [contentLoading, setContentLoading] = React.useState(false);

  const handleLanguageChange = (newLanguage: "vi" | "en") => {
    setContentLoading(true);
    setLanguage(newLanguage);
    setTimeout(() => {
      setContentLoading(false);
    }, 800);
  };

  const LanguageSelectorProps = {
    language,
    setLanguage: handleLanguageChange,
  };

  useEffect(() => {
    const timer = setTimeout(() => {
      setLoading(false);
    }, 1000);

    return () => clearTimeout(timer);
  }, []);

  const content = {
    vi: {
      quickRecap: {
        title: "Tổng quan nhanh",
        text: "SignMeet là ứng dụng hỗ trợ phụ đề thời gian thực và dịch đa ngôn ngữ, giúp giao tiếp trở nên dễ dàng cho mọi người. Tính năng nổi bật nhất là khả năng hỗ trợ người khiếm thính bằng cách chuyển đổi giọng nói thành ngôn ngữ ký hiệu. Trong buổi demo, Lam và Duc Anh đã thực hiện một cuộc hội thoại ngắn bằng tiếng Anh, cho thấy khả năng chuyển đổi thành công sang ngôn ngữ ký hiệu Mỹ (ASL).",
      },
      nextSteps: {
        title: "Các bước tiếp theo",
        items: [
          "Tiếp tục phát triển và hoàn thiện tính năng mô phỏng ngôn ngữ ký hiệu",
          "Tối ưu hóa độ trễ của hệ thống phụ đề và dịch thuật xuống dưới 300ms",
          "Mở rộng hỗ trợ thêm nhiều ngôn ngữ và phương ngữ ký hiệu khác nhau",
          "Tiến hành thử nghiệm với cộng đồng người khiếm thính để nhận phản hồi",
        ],
      },
      summary: {
        title: "Tóm tắt",
        sections: {
          mainFeatures: {
            title: "Tính năng chính",
            content:
              "Ứng dụng cung cấp phụ đề thời gian thực và hỗ trợ đa ngôn ngữ, bao gồm tiếng Việt, tiếng Đức và nhiều ngôn ngữ khác.",
          },
          demoFeatures: {
            title: "Demo tính năng",
            content:
              "Trình diễn thành công việc chuyển đổi cuộc hội thoại tiếng Anh sang ngôn ngữ ký hiệu ASL trong thời gian thực.",
          },
          technicalDetails: {
            title: "Chi tiết kỹ thuật",
            content:
              "Tích hợp công nghệ chuyển đổi giọng nói thành ngôn ngữ ký hiệu và hệ thống phụ đề đa ngôn ngữ thời gian thực.",
          },
          conclusion: {
            title: "Kết luận",
            content:
              "SignMeet thể hiện khả năng vượt trội trong việc hỗ trợ giao tiếp cho người khiếm thính và phá vỡ rào cản ngôn ngữ.",
          },
        },
      },
    },
    en: {
      quickRecap: {
        title: "Quick Recap",
        text: "SignMeet is an application that supports real-time subtitles and multi-language translation, making communication seamless for everyone. The most outstanding feature is its ability to support deaf people by translating voice into sign language. During the demo, Thanh and Duc Anh conducted a short conversation in English, demonstrating successful conversion to American Sign Language (ASL).",
      },
      nextSteps: {
        title: "Next Steps",
        items: [
          "Continue developing and refining sign language simulation features",
          "Optimize subtitle and translation system latency to under 300ms",
          "Expand support for additional languages and sign language dialects",
          "Conduct testing with the deaf community for feedback",
        ],
      },
      summary: {
        title: "Summary",
        sections: {
          mainFeatures: {
            title: "Main Features",
            content:
              "Application provides real-time subtitles and multi-language support, including Vietnamese, German, and many other languages.",
          },
          demoFeatures: {
            title: "Demo Features",
            content:
              "Successfully demonstrated real-time conversion of English conversation to ASL sign language.",
          },
          technicalDetails: {
            title: "Technical Details",
            content:
              "Integrates voice-to-sign-language conversion technology and real-time multilingual subtitle system.",
          },
          conclusion: {
            title: "Conclusion",
            content:
              "SignMeet demonstrates exceptional capability in supporting communication for the deaf and breaking down language barriers.",
          },
        },
      },
    },
  };

  return (
    <div className="p-4 sm:p-8 bg-gray-50 min-h-screen">
      <div className="max-w-5xl mx-auto bg-white shadow-lg rounded-lg border border-gray-200">
        <div className="p-4 sm:p-6 border-b border-gray-300 relative">
          <Image
            src={"images/gemini.svg"}
            width={25}
            height={25}
            alt={"Gemini"}
            className={`absolute top-4 left-4 ${
              loading ? "rotating-image" : ""
            }`}
          />
          <div className="absolute top-4 right-4">
            <LanguageSelector {...LanguageSelectorProps} />
          </div>

          {loading ? (
            <>
              <Skeleton
                variant="rectangular"
                width={700}
                height={18}
                sx={{
                  background:
                    "linear-gradient(90deg, #ADD8E6, #FFC0CB, #ADD8E6)",
                  opacity: 0.8,
                  backgroundSize: "200% 100%",
                  animation: "gradient-loading 1.5s infinite",
                  borderRadius: 2, // Optional: Rounded corners
                }}
              />
              <Skeleton
                variant="rectangular"
                className="mt-4"
                width={700}
                height={18}
                sx={{
                  background:
                    "linear-gradient(90deg, #ADD8E6, #FFC0CB, #ADD8E6)",
                  opacity: 0.8,
                  backgroundSize: "200% 100%",
                  animation: "gradient-loading 1.5s infinite",
                  borderRadius: 2, // Optional: Rounded corners
                }}
              />
            </>
          ) : (
            <div className="text-center pt-8">
              <h1 className="text-2xl font-bold text-gray-800 animate-fade-in">
                Meeting Summary
              </h1>
              <p className="text-gray-600 text-sm mt-1 animate-fade-in">
                Mar 12, 2023 · 09:13 AM Pacific Time (US and Canada) · ID: 142
                534 5235
              </p>
            </div>
          )}
        </div>

        {/* Quick Recap */}
        <div className="p-4 sm:p-6 border-b border-gray-300">
          {loading || contentLoading ? (
            <>
              <Skeleton
                variant="rectangular"
                width={700}
                height={18}
                sx={{
                  background:
                    "linear-gradient(90deg, #ADD8E6, #FFC0CB, #ADD8E6)",
                  opacity: 0.8,
                  backgroundSize: "200% 100%",
                  animation: "gradient-loading 1.5s infinite",
                  borderRadius: 2, // Optional: Rounded corners
                }}
              />
              <Skeleton
                variant="rectangular"
                className="mt-4"
                width={700}
                height={60}
                sx={{
                  background:
                    "linear-gradient(90deg, #ADD8E6, #FFC0CB, #ADD8E6)",
                  opacity: 0.8,
                  backgroundSize: "200% 100%",
                  animation: "gradient-loading 1.5s infinite",
                  borderRadius: 2, // Optional: Rounded corners
                }}
              />
            </>
          ) : (
            <>
              <h2 className="text-lg font-semibold mb-2 text-gray-800 animate-fade-in">
                {content[language].quickRecap.title}
              </h2>
              <p className="text-gray-700 animate-fade-in">
                {content[language].quickRecap.text}
              </p>
            </>
          )}
        </div>

        {/* Next Steps */}
        {/* <div className="p-4 sm:p-6 border-b border-gray-300">
          {loading || contentLoading ? (
            <>
              <Skeleton
                variant="rectangular"
                width={700}
                height={18}
                sx={{
                  background:
                    "linear-gradient(90deg, #ADD8E6, #FFC0CB, #ADD8E6)",
                  opacity: 0.8,
                  backgroundSize: "200% 100%",
                  animation: "gradient-loading 1.5s infinite",
                  borderRadius: 2, // Optional: Rounded corners
                }}
              />
              <Skeleton
                variant="rectangular"
                className="mt-4"
                width={700}
                height={60}
                sx={{
                  background:
                    "linear-gradient(90deg, #ADD8E6, #FFC0CB, #ADD8E6)",
                  opacity: 0.8,
                  backgroundSize: "200% 100%",
                  animation: "gradient-loading 1.5s infinite",
                  borderRadius: 2, // Optional: Rounded corners
                }}
              />
            </>
          ) : (
            <>
              <h2 className="text-lg font-semibold mb-2 text-gray-800 animate-fade-in">
                {content[language].nextSteps.title}
              </h2>
              <ul className="list-disc list-inside text-gray-700 space-y-2 animate-fade-in">
                {content[language].nextSteps.items.map((item, index) => (
                  <li key={index}>{item}</li>
                ))}
              </ul>
            </>
          )}
        </div> */}

        {/* Summary */}
        <div className="p-4 sm:p-6">
          {loading || contentLoading ? (
            <>
              <Skeleton
                variant="rectangular"
                width={700}
                height={18}
                sx={{
                  background:
                    "linear-gradient(90deg, #ADD8E6, #FFC0CB, #ADD8E6)",
                  opacity: 0.8,
                  backgroundSize: "200% 100%",
                  animation: "gradient-loading 1.5s infinite",
                  borderRadius: 2, // Optional: Rounded corners
                }}
              />
              <Skeleton
                variant="rectangular"
                className="mt-4"
                width={700}
                height={500}
                sx={{
                  background:
                    "linear-gradient(90deg, #ADD8E6, #FFC0CB, #ADD8E6)",
                  opacity: 0.8,
                  backgroundSize: "200% 100%",
                  animation: "gradient-loading 1.5s infinite",
                  borderRadius: 2, // Optional: Rounded corners
                }}
              />
            </>
          ) : (
            <>
              <h2 className="text-lg font-semibold mb-2 text-gray-800 animate-fade-in">
                {content[language].summary.title}
              </h2>
              <div className="space-y-4 text-gray-700 animate-fade-in">
                {Object.entries(content[language].summary.sections).map(
                  ([key, section]) => (
                    <div key={key}>
                      <h3 className="font-medium text-gray-800">
                        {section.title}
                      </h3>
                      <p>{section.content}</p>
                    </div>
                  )
                )}
              </div>
            </>
          )}
        </div>

        {/* Footer with Actions */}
        <div className="flex flex-col sm:flex-row justify-between items-center p-4 sm:p-6 bg-gray-100 border-t border-gray-200 gap-4">
          <button
            onClick={() => router.push("/home")}
            className="w-full sm:w-auto flex items-center justify-center gap-2 px-4 py-2 bg-gray-200 hover:bg-gray-300 text-gray-700 rounded-lg transition-transform transform hover:scale-105"
          >
            <FiHome />
            Back to Home
          </button>
          <div className="flex flex-wrap sm:flex-nowrap gap-2 sm:gap-4 w-full sm:w-auto justify-center">
            <button className="flex items-center gap-2 px-4 py-2 bg-blue-500 hover:bg-blue-600 text-white rounded-lg transition-transform transform hover:scale-105">
              <FiShare2 />
              Share
            </button>
            <button className="flex items-center gap-2 px-4 py-2 bg-gray-200 hover:bg-gray-300 text-gray-700 rounded-lg transition-transform transform hover:scale-105">
              <FiDownload />
              Download
            </button>
            <button className="flex items-center gap-2 px-4 py-2 bg-gray-200 hover:bg-gray-300 text-gray-700 rounded-lg transition-transform transform hover:scale-105">
              <FiEdit />
              Edit
            </button>
            <button className="flex items-center gap-2 px-4 py-2 bg-red-500 hover:bg-red-600 text-white rounded-lg transition-transform transform hover:scale-105">
              <FiTrash />
              Delete
            </button>
          </div>
        </div>
      </div>
    </div>
  );
};

export default MeetingSummary;
