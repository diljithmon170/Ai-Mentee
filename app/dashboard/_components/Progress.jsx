import React from "react";

const Progress = ({ value, max }) => {
  return (
    <div className="w-full bg-gray-300 rounded-full h-4 my-3">
      <div
        className="bg-blue-500 h-4 rounded-full"
        style={{ width: `${(value / max) * 100}%` }}
      ></div>
    </div>
  );
};

export default Progress;
