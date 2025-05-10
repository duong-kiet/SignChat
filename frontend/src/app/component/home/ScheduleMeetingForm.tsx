"use client";
import React, { useState } from "react";

const ScheduleMeetingForm = () => {
  const [topic, setTopic] = useState("");
  const [date, setDate] = useState("2024-11-19");
  const [startTime, setStartTime] = useState("20:30");
  const [endTime, setEndTime] = useState("21:00");
  const [timeZone, setTimeZone] = useState("Bangkok");
  const [isRecurring, setIsRecurring] = useState(false);
  const [attendees, setAttendees] = useState("");
  const [meetingId, setMeetingId] = useState(true);
  const [passcode, setPasscode] = useState("0H9BNZ");
  const [waitingRoom, setWaitingRoom] = useState(true);

  const handleSubmit = (e: any) => {
    e.preventDefault();
    // Handle form submission
  };

  return (
    <>
      <h1 className="font-bold text-[48px] mb-4">Schedule Meeting</h1>
      <form onSubmit={handleSubmit} className="space-y-4">
        <div>
          <label htmlFor="topic" className="block font-medium mb-1">
            Topic
          </label>
          <input
            type="text"
            id="topic"
            value={topic}
            onChange={(e) => setTopic(e.target.value)}
            className="border rounded-md px-3 py-2 w-full"
          />
        </div>

        <div className="flex space-x-4">
          <div className="flex-1">
            <label htmlFor="date" className="block font-medium mb-1">
              Date
            </label>
            <input
              type="date"
              id="date"
              value={date}
              onChange={(e) => setDate(e.target.value)}
              className="border rounded-md px-3 py-2 w-full"
            />
          </div>
          <div className="flex-1">
            <label htmlFor="start-time" className="block font-medium mb-1">
              Start Time
            </label>
            <input
              type="time"
              id="start-time"
              value={startTime}
              onChange={(e) => setStartTime(e.target.value)}
              className="border rounded-md px-3 py-2 w-full"
            />
          </div>
          <div className="flex-1">
            <label htmlFor="end-time" className="block font-medium mb-1">
              End Time
            </label>
            <input
              type="time"
              id="end-time"
              value={endTime}
              onChange={(e) => setEndTime(e.target.value)}
              className="border rounded-md px-3 py-2 w-full"
            />
          </div>
        </div>

        <div>
          <label htmlFor="time-zone" className="block font-medium mb-1">
            Time Zone
          </label>
          <select
            id="time-zone"
            value={timeZone}
            onChange={(e) => setTimeZone(e.target.value)}
            className="border rounded-md px-3 py-2 w-full"
          >
            <option value="Bangkok">Bangkok</option>
            {/* Add more time zone options as needed */}
          </select>
        </div>
        <div>
          <label className="inline-flex items-center">
            <input
              type="checkbox"
              checked={isRecurring}
              onChange={(e) => setIsRecurring(e.target.checked)}
              className="rounded text-blue-500 focus:ring-blue-500"
            />
            <span className="ml-2 font-medium">Recurring meeting</span>
          </label>
        </div>
        <div>
          <label htmlFor="attendees" className="block font-medium mb-1">
            Attendees
          </label>
          <input
            type="text"
            placeholder="Name or email address"
            id="attendees"
            value={attendees}
            onChange={(e) => setAttendees(e.target.value)}
            className="border rounded-md px-3 py-2 w-full"
          />
        </div>

        <div>
          <label className="block font-medium mb-4">Meeting ID</label>
          <div>
            <label className="inline-flex items-center">
              <input
                type="radio"
                name="meeting-id"
                checked={meetingId}
                onChange={() => setMeetingId(true)}
                className="rounded-full text-blue-500 focus:ring-blue-500"
              />
              <span className="ml-2">Generate Automatically</span>
            </label>
            <label className="inline-flex items-center ml-4">
              <input
                type="radio"
                name="meeting-id"
                checked={!meetingId}
                onChange={() => setMeetingId(false)}
                className="rounded-full text-blue-500 focus:ring-blue-500"
              />
              <span className="ml-2">Personal Meeting ID 242 292 8464</span>
            </label>
          </div>
        </div>

        <div>
          <label htmlFor="passcode" className="block font-medium mb-1">
            Passcode
          </label>
          <input
            type="text"
            id="passcode"
            value={passcode}
            onChange={(e) => setPasscode(e.target.value)}
            className="border rounded-md px-3 py-2 w-full"
          />
          <p className="text-sm text-gray-500 mt-1">
            Only users who have the invite link or passcode can join the meeting
          </p>
        </div>

        <div>
          <label className="inline-flex items-center">
            <input
              type="checkbox"
              checked={waitingRoom}
              onChange={(e) => setWaitingRoom(e.target.checked)}
              className="rounded text-blue-500 focus:ring-blue-500"
            />
            <span className="ml-2 font-medium">Waiting Room</span>
          </label>
          <p className="text-sm text-gray-500 mt-1">
            Only users admitted by the host can join the meeting
          </p>
        </div>

        <button
          type="submit"
          className="bg-blue-500 hover:bg-blue-600 text-white font-medium px-4 py-2 rounded-md"
        >
          Save
        </button>
      </form>
    </>
  );
};

export default ScheduleMeetingForm;
