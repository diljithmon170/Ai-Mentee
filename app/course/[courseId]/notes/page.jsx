"use client";

import { useParams } from "next/navigation";
import axios from "axios";
import React, { useEffect, useState } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";

function ViewNotes() {
  const { courseId } = useParams();
  const [notes, setNotes] = useState([]);
  const [error, setError] = useState(null);
  const [loading, setLoading] = useState(true);
  const [stepCount, setStepCount] = useState(0);

  const prevStep = () => stepCount > 0 && setStepCount(stepCount - 1);
  const nextStep = () =>
    stepCount < notes.length - 1 && setStepCount(stepCount + 1);

  useEffect(() => {
    if (courseId) {
      fetchNotes();
    }
  }, [courseId]);

  const fetchNotes = async () => {
    try {
      const result = await axios.post("/api/study-type", {
        courseId: courseId,
        studyType: "notes",
      });
      console.log("NOTES", result.data);
      setNotes(result.data);
    } catch (err) {
      console.error("Error fetching notes:", err.message);
      setError("Failed to fetch notes.");
    } finally {
      setLoading(false);
    }
  };

  if (loading)
    return <div className="text-center text-blue-500">Loading notes...</div>;
  if (error) return <div className="text-center text-red-500">{error}</div>;

  let jsonObject = null;
  try {
    if (!notes || !notes[stepCount] || !notes[stepCount].notes) {
      throw new Error("Invalid notes data");
    }
    const jsonString = notes[stepCount].notes;
    console.log(jsonString);
    jsonObject = JSON.parse(jsonString);
    console.log("Content", jsonObject);
  } catch (err) {
    console.error("Error parsing JSON:", err.message);
    return (
      <div className="text-center text-red-500">
        Error: Failed to parse notes data.
      </div>
    );
  }

  return (
    <div className="h-screen p-5">
      {/* Navigation */}
      <div className="flex gap-5 items-center mb-5">
        <button
          className="btn btn-outline-primary"
          onClick={prevStep}
          disabled={stepCount === 0 || notes.length === 0}
        >
          Previous
        </button>

        <div className="flex w-full gap-2">
          {notes.map((_, index) => (
            <div
              key={index}
              className={`w-full h-2 rounded-full ${
                index <= stepCount ? "bg-primary" : "bg-gray-300"
              }`}
            ></div>
          ))}
        </div>

        <button
          className="btn btn-outline-primary"
          onClick={nextStep}
          disabled={stepCount === notes.length - 1 || notes.length === 0}
        >
          Next
        </button>
      </div>

      {/* Render Content */}
      {jsonObject && (
        <div>
          <div className="flex text-2xl font-bold mb-3">
            <span className="pr-3">{jsonObject.emoji}</span>
            {jsonObject.chapterTitle}
          </div>
          <p className="text-gray-700 mb-5">{jsonObject.chapterSummary}</p>

          {jsonObject.topics.map((topic, index) => (
            <div
              key={index}
              className="p-4 bg-gray-100 rounded-lg shadow-md mb-4"
            >
              <h1 className="text-lg font-bold mb-2">{topic.topicTitle}</h1>
              {/* Render Markdown Content */}
              <ReactMarkdown
                children={topic.content}
                remarkPlugins={[remarkGfm]}
              />
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

export default ViewNotes;
