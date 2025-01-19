"use client"
import React, { useState } from 'react'
import SelectOption from './_components/SelectOption'
import TopicInput from './_components/TopicInput';
import axios from 'axios';
import { v4 as uuidv4 } from 'uuid';
import { useUser } from '@clerk/nextjs';
import { Loader } from 'lucide-react';
import { useRouter } from 'next/navigation';
import Alert from '../dashboard/_components/Alert';

function Create() {
    const [step, setStep] = useState(0);
    const [formData, setFormData] = useState([])
    const [loading, setLoading] = useState(false);
    const router = useRouter();
    const {user}=useUser();
    const handleUserInput=(fieldName, fieldValue)=>{
        setFormData(prev=>({
            ...prev,
            [fieldName]: fieldValue
        }))
    }
    
    console.log("Form Data:", formData);
    console.log("User Email:", user?.primaryEmailAddress?.emailAddress);

    const GenerateCourseOutline = async () => {
      try {
        const courseId = uuidv4();
        setLoading(true);
        const payload = {
          courseId,
          courseType: formData.studyType,
          topic: formData.topic,
          difficultyLevel: formData.difficultyLevel || "Medium",
          createdBy: user?.primaryEmailAddress?.emailAddress,
        };

        console.log("Payload being sent:", payload);

        const result = await axios.post(
          "/api/generate-course-outline",
          payload
        );

        console.log("API Response:", result.data.result.resp);
        setLoading(false);
        router.replace('/dashboard')
      } catch (error) {
        console.error(
          "Error generating course outline:",
          error?.response?.data || error.message
        );
        alert("Failed to generate course outline. Please check your input.");
      }
    };

  return (
    <div className='flex flex-col items-center p-5 md:px-24 lg:px-36 h-screen'>
        <h2 className='font-bold text-4xl text-primary'>Start Building Your Personal Study Material</h2>
        <p className='text-gray-500 text-lg'>Fill all the details in order to generate study material for your next project</p>
        <div className='mt-10'>
            {step==0? <SelectOption selectedStudyType={(value)=>handleUserInput("studyType",value)}/> : <TopicInput setTopic={(value)=>handleUserInput("topic", value)} setDifficultyLevel={(value)=>handleUserInput("difficultyLevel", value)}/>}
        </div>
        <div className="flex justify-between w-full mt-32">
        {step!=0? <button className="btn btn-primary" onClick={()=>setStep(step-1)}>Previous</button>:'-'}
        {step==0? <button onClick={()=>setStep(step+1)} className="btn btn-outline-primary">Next</button> : <button className="btn btn-outline-primary" onClick={GenerateCourseOutline} disabled={loading}>{loading?<Loader className='animate-spin'/>:"Generate"}</button>}
      </div>
    </div>
  )
}

export default Create