"use client";

import axios from "axios";
import { useParams } from "next/navigation";
import React, { useEffect, useState } from "react";

function GamifiedQuiz() {
  const { courseId } = useParams();
  const [stepCount, setStepCount] = useState(0);
  const [quizData, setQuizData] = useState(null);
  const [loading, setLoading] = useState(true);
  const [selectedOption, setSelectedOption] = useState(null);
  const [selectedOptions, setSelectedOptions] = useState([]);
  const [isCorrect, setIsCorrect] = useState(null);
  const [score, setScore] = useState(0);
  const [timer, setTimer] = useState(15);
  const [quizCompleted, setQuizCompleted] = useState(false);

  useEffect(() => {
    GetQuiz();
  }, []);

  useEffect(() => {
    if (timer > 0 && selectedOption === null && !quizCompleted) {
      const countdown = setInterval(() => setTimer((t) => t - 1), 1000);
      return () => clearInterval(countdown);
    } else if (timer === 0 && !quizCompleted) {
      handleTimeout();
    }
  }, [timer, selectedOption, quizCompleted]);

  const GetQuiz = async () => {
    try {
      const result = await axios.post("/api/study-type", {
        courseId: courseId,
        studyType: "Quiz",
      });
      setQuizData(result.data);
      setSelectedOptions(
        Array(result.data.content.questions.length).fill(null)
      );
      setLoading(false);
    } catch (error) {
      console.error("Error fetching quiz data:", error);
      setLoading(false);
    }
  };

  const handleOptionClick = (option) => {
    setSelectedOption(option);

    const correctAnswer = quizData.content.questions[stepCount].answer;
    const isAnswerCorrect = option === correctAnswer;

    setIsCorrect(isAnswerCorrect);

    if (isAnswerCorrect) {
      setScore((prev) => prev + 10);
    }

    // Save selected option
    const updatedSelectedOptions = [...selectedOptions];
    updatedSelectedOptions[stepCount] = option;
    setSelectedOptions(updatedSelectedOptions);
  };

  const handleTimeout = () => {
    setSelectedOption(null);
    setIsCorrect(false);

    // Save timeout state as null
    const updatedSelectedOptions = [...selectedOptions];
    updatedSelectedOptions[stepCount] = null;
    setSelectedOptions(updatedSelectedOptions);
  };

  const resetSelection = () => {
    setTimer(15);
    setSelectedOption(null);
    setIsCorrect(null);
  };

  const previousStep = () => {
    if (stepCount > 0) {
      setStepCount((prev) => prev - 1);
      resetSelection();
      // Restore the previously selected option for the question
      const prevOption = selectedOptions[stepCount - 1];
      setSelectedOption(prevOption);
      setIsCorrect(
        prevOption === quizData.content.questions[stepCount - 1].answer
      );
    }
  };

  const nextStep = () => {
    if (quizData && quizData.content.questions.length > stepCount + 1) {
      setStepCount((prev) => prev + 1);
      resetSelection();
      // Restore the previously selected option for the question
      const nextOption = selectedOptions[stepCount + 1];
      setSelectedOption(nextOption);
      setIsCorrect(
        nextOption === quizData.content.questions[stepCount + 1].answer
      );
    } else {
      setQuizCompleted(true); // Mark quiz as completed
    }
  };

  const restartQuiz = () => {
    setStepCount(0);
    setScore(0);
    setSelectedOptions(Array(quizData.content.questions.length).fill(null));
    setQuizCompleted(false);
    resetSelection();
  };

  if (loading) {
    return (
      <div className="flex justify-center items-center h-screen">
        <div className="text-center">
          <div className="animate-spin w-10 h-10 border-4 border-primary rounded-full border-t-transparent"></div>
          <p>Loading Quiz...</p>
        </div>
      </div>
    );
  }

  if (!quizData || !quizData.content || !quizData.content.questions.length) {
    return (
      <div className="flex justify-center items-center h-screen">
        <p>No Quiz Data Available</p>
      </div>
    );
  }

  const { questions, quizTitle } = quizData.content;

  if (quizCompleted) {
    return (
      <div className="h-screen flex items-center justify-center">
        <div className="bg-white shadow-lg rounded-lg p-8 animate-fade-in">
          <h1 className="text-3xl font-bold mb-4 text-center">
            Quiz Completed! 🎉
          </h1>
          <p className="text-lg mb-2 text-center">
            You scored <span className="text-green-500 font-bold">{score}</span>{" "}
            out of {questions.length * 10}.
          </p>
          <div className="text-center">
            <button className="btn btn-primary mt-5" onClick={restartQuiz}>
              Restart Quiz
            </button>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="h-screen p-5">
      <h1 className="text-3xl font-bold mb-5 text-center">{quizTitle}</h1>

      <div className="mb-5">
        <div className="flex justify-between items-center">
          <p className="text-lg font-medium">
            Question {stepCount + 1} of {questions.length}
          </p>
          <p className="text-lg font-medium text-red-500">Time: {timer}s</p>
          <p className="text-lg font-medium text-green-500">Score: {score}</p>
        </div>

        <div className="mt-16 mb-10">
          <p className="text-black text-2xl text-center">
            {questions[stepCount].question}
          </p>
        </div>

        <div className="flex justify-center flex-wrap">
          <div className="grid grid-cols-1 gap-3">
            {questions[stepCount].options.map((option, index) => (
              <button
                key={index}
                onClick={() => handleOptionClick(option)}
                className={`btn text-[16px] ${
                  selectedOption === option
                    ? isCorrect
                      ? "btn-success"
                      : "btn-error"
                    : selectedOption !== null &&
                      option === questions[stepCount].answer
                    ? "btn-success"
                    : "btn-outline-primary"
                }`}
                disabled={selectedOption !== null || timer === 0}
              >
                {option}
              </button>
            ))}
          </div>
        </div>

        {selectedOption && (
          <p
            className={`mt-3 ${
              isCorrect ? "text-green-500" : "text-red-500"
            } font-bold`}
          >
            {isCorrect
              ? `Correct! 🎉`
              : `Incorrect. The correct answer is: ${questions[stepCount].answer}`}
          </p>
        )}
      </div>

      <div className="flex gap-5 items-center justify-between">
        <button
          className="btn btn-outline-secondary"
          onClick={previousStep}
          disabled={stepCount === 0}
        >
          Previous
        </button>
        <button
          className="btn btn-outline-primary"
          onClick={nextStep}
          disabled={quizCompleted}
        >
          {stepCount === questions.length - 1 ? "Finish" : "Next"}
        </button>
      </div>
    </div>
  );
}

export default GamifiedQuiz;
