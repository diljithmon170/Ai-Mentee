"use client";

import { useParams } from "next/navigation";
import React, { useEffect, useState } from "react";
import DashboardHeader from "../../dashboard/_components/DashboardHeader";
import axios from "axios";
import CourseIntroCard from "./_components/CourseIntroCard";
import StudyMaterialSection from "./_components/StudyMaterialSection";
import ChapterList from "./_components/ChapterList";

function Course() {
  const { courseId } = useParams();
  const [courseData, setCourseData] = useState(null);
  const [error, setError] = useState(null);

  const GetCourse = async () => {
    try {
      const result = await axios.get(`/api/courses?courseId=${courseId}`);
      console.log(result.data.result); // Log the response
      setCourseData(result.data.result);
    } catch (err) {
      console.error("Error fetching course:", err.message);
      setError("Failed to load course data.");
    }
  };

  useEffect(() => {
    if (courseId) {
      GetCourse();
    }
  }, [courseId]);

  if (error) {
    return <div>Error: {error}</div>;
  }

  if (!courseData) {
    return <div>Loading course details...</div>;
  }

  return (
    <div className="h-screen">
      <div>
        <CourseIntroCard course={courseData} />
        <StudyMaterialSection courseId={courseId} course={courseData}/>
        <ChapterList course={courseData} />
      </div>
    </div>
  );
}

export default Course;
