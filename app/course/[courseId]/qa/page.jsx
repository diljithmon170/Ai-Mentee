"use client";

import axios from "axios";
import { useParams } from "next/navigation";
import React, { useEffect, useState } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";

function ViewQA() {
  const [qaData, setQaData] = useState([]);
  const [stepCount, setStepCount] = useState(0);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [showAnswer, setShowAnswer] = useState(true); // Tracks whether the answer is shown
  const { courseId } = useParams();

  useEffect(() => {
    GetQA();
  }, [courseId]);

  const GetQA = async () => {
    try {
      const result = await axios.post("/api/study-type", {
        courseId: courseId,
        studyType: "Question/Answer",
      });
      console.log("QA DATA", result.data);
      setQaData(result.data.content.questions);
    } catch (err) {
      console.error("Error fetching QA data:", err.message);
      setError("Failed to fetch QA data.");
    } finally {
      setLoading(false);
    }
  };

  const prevStep = () => {
    setStepCount((prev) => (prev > 0 ? prev - 1 : prev));
  };

  const nextStep = () => {
    setStepCount((prev) => (prev < qaData.length - 1 ? prev + 1 : prev));
  };

  const toggleAnswer = () => {
    setShowAnswer((prev) => !prev);
  };

  if (loading)
    return (
      <div className="text-center text-blue-500">Loading questions...</div>
    );
  if (error) return <div className="text-center text-red-500">{error}</div>;

  return (
    <div className="h-screen p-5">
      {/* Navigation */}
      <div className="flex gap-5 items-center mb-5">
        <button
          className="btn btn-outline-primary"
          onClick={prevStep}
          disabled={stepCount === 0 || qaData.length === 0}
        >
          Previous
        </button>

        <div className="flex w-full gap-2">
          {qaData.map((_, index) => (
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
          disabled={stepCount === qaData.length - 1 || qaData.length === 0}
        >
          Next
        </button>
      </div>

      {/* Render QA Content */}
      {qaData[stepCount] && (
        <div>
          <h1 className="text-2xl font-bold mb-3">
            {qaData[stepCount].question}
          </h1>
          <button
            className="btn btn-outline-secondary mb-3"
            onClick={toggleAnswer}
          >
            {showAnswer ? "Hide Answer" : "Show Answer"}
          </button>

          {/* Smoothly toggle the answer inside a container */}
          <div
            className={`transition-all duration-500 ease-in-out overflow-hidden bg-white p-4 rounded-md shadow-md`}
            style={{
              maxHeight: showAnswer ? "1000px" : "0", // Transition max height
              opacity: showAnswer ? "1" : "0", // Transition opacity
            }}
          >
            <div
              className={`transition-all duration-500 ease-in-out`}
              style={{
                opacity: showAnswer ? "1" : "0", // Smooth fade-in effect
                maxHeight: showAnswer ? "1000px" : "0", // Smooth expand/collapse
              }}
            >
              {/* Render Markdown Content */}
              <ReactMarkdown
                children={qaData[stepCount].answer}
                remarkPlugins={[remarkGfm]}
              />
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

export default ViewQA;
