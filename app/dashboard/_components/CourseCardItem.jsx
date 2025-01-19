import React from "react";
import Loader from "./Loader";
import Link from "next/link";

function CourseCardItem({ course }) {
  console.log("course", course.status);

  return (
    <div className="p-4  w-full border rounded-lg shadow-md bg-gray-100">
      <div className="flex flex-col justify-between">
        <div className="flex items-center mb-4">
          <img
            src="/knowledge.png"
            alt="knowledge"
            width={50}
            height={50}
            className="mr-4"
          />
          <h2 className="text-lg font-semibold">
            {course.courseLayout.courseTitle}
          </h2>
        </div>

        <div className="text-sm mt-3  text-gray-600 bg-[#ededed] py-5 px-3 rounded-lg">
          <p className="line-clamp-4">{course.courseLayout.courseSummary}</p>
        </div>

        <div className="mt-5 flex justify-end items-center">
          {course.status === "Generating" ? (
            <div className="flex justify-center items-center">
              <Loader />
            </div>
          ) : (
            <Link href={`course/${course.courseId}`}>
              <button className="btn btn-outline-primary px-4 py-2">
                View
              </button>
            </Link>
          )}
        </div>
      </div>
    </div>
  );
}

export default CourseCardItem;
