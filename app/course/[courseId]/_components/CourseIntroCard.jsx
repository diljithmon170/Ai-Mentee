import React from "react";

function CourseIntroCard({ course }) {
  return (
    <div className="flex gap-6 items-center p-6 border shadow-md rounded-lg bg-gray-100 w-full mx-auto">

      <div className="w-[120px] flex flex-col justify-center items-center">
        <img
          src="/knowledge.png"
          alt="knowledge"
          width={70}
          height={70}
          className="mb-4"
        />
        <h2 className="text-xs text-blue-600 font-semibold">
          Total Chapters: {course.courseLayout.chapters.length}
        </h2>
      </div>

      <div className="flex flex-col justify-between">
        <h2 className="font-bold text-xl text-gray-800">
          {course.courseLayout.courseTitle}
        </h2>
        <p className="text-gray-600 text-sm mt-2">
          {course.courseLayout.courseSummary}
        </p>
      </div>
    </div>
  );
}

export default CourseIntroCard;
