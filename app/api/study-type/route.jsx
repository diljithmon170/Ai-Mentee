import { CHAPTER_NOTES_TABLE } from "/configs/schema";
import { db } from "/configs/db";
import { eq, and } from "drizzle-orm";
import { NextResponse } from "next/server";
import { STUDY_TYPE_CONTENT_TABLE } from "../../../configs/schema";

export async function POST(req) {
  try {
    const { courseId, studyType } = await req.json();

    // Validate input
    if (!courseId) {
      return NextResponse.json(
        { error: "The 'courseId' field is required." },
        { status: 400 }
      );
    }

    console.log("Incoming Data:", { courseId, studyType });

    // Handling "ALL" case
    if (studyType === "ALL") {
      const notes = await db
        .select()
        .from(CHAPTER_NOTES_TABLE)
        .where(eq(CHAPTER_NOTES_TABLE.courseId, courseId));

      console.log("Fetched Notes:", notes);

      const contentList = await db
        .select()
        .from(STUDY_TYPE_CONTENT_TABLE)
        .where(eq(STUDY_TYPE_CONTENT_TABLE.courseId, courseId));

      console.log("Fetched Study Content:", contentList);

      const result = {
        notes: notes,
        flashcard: contentList.filter((item) => item.type === "Flashcard"),
        quiz: contentList.filter((item) => item.type === "Quiz"),
        qa: contentList.filter((item) => item.type === "Question/Answer"),
      };

      return NextResponse.json(result);
    }

    // Handling specific study types
    else if (studyType === "notes") {
      const notes = await db
        .select()
        .from(CHAPTER_NOTES_TABLE)
        .where(eq(CHAPTER_NOTES_TABLE.courseId, courseId));

      console.log("Notes for courseId:", courseId, notes);

      return NextResponse.json(notes);
    } else {
      const result = await db
        .select()
        .from(STUDY_TYPE_CONTENT_TABLE)
        .where(
          and(
            eq(STUDY_TYPE_CONTENT_TABLE.courseId, courseId),
            eq(STUDY_TYPE_CONTENT_TABLE.type, studyType)
          )
        );

      console.log(`Content for type ${studyType}:`, result);

      return NextResponse.json(result[0]);
    }
  } catch (error) {
    console.error("Error in POST /api/study-type:", error.message);
    console.error("Full error details:", error); // Log full error details

    return NextResponse.json(
      { error: "Failed to fetch study materials. Please try again later." },
      { status: 500 }
    );
  }
}
