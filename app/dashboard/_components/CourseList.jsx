"use client";

import React, { useEffect, useState } from "react";
import { useUser } from "@clerk/nextjs";
import axios from "axios";
import CourseCardItem from "./CourseCardItem";

function CourseList() {
  const { user } = useUser();
  const [courseList, setCourseList] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  const GetCourseList = async () => {
    try {
      const result = await axios.post("/api/courses/", {
        createdBy: user?.primaryEmailAddress?.emailAddress,
      });
      setCourseList(result.data.result);
      console.log(result.data.result) // Assuming the API returns `courses` array
    } catch (err) {
      console.error("Error fetching course list:", err.message);
      setError("Failed to load courses. Please try again later.");
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    if (user) {
      GetCourseList();
    }
  }, [user]);

  if (loading) return <div>Loading...</div>;
  if (error) return <div>{error}</div>;

  return (
    <div className="mt-10">
        <h2 className="font-bold text-2xl">Your Study Material</h2>
        <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-4 mt-2 gap-5">
            {courseList.map((course, index) => (
                <CourseCardItem course={course} key={index}/>
            ))}
        </div>
    </div>
  );
}

export default CourseList;
