import {
  CHAPTER_NOTES_TABLE,
  STUDY_MATERIAL_TABLE,
  STUDY_TYPE_CONTENT_TABLE,
  USER_TABLE,
} from "../configs/schema";
import { db } from "../configs/db";
import { inngest } from "./client";
import {
  generateNotesAiModel,
  GenerateQaAiModel,
  GenerateQuizAiModel,
  GenerateStudyTypeContentAiModel,
} from "../configs/AiModel";
import { eq } from "drizzle-orm";

// Function to test the "hello-world" event
export const helloWorld = inngest.createFunction(
  { id: "hello-world" },
  { event: "test/hello.world" },
  async ({ event, step }) => {
    await step.sleep("wait-a-moment", "1s");
    return { message: `Hello ${event.data.email}!` };
  }
);

// Function to create a new user if they don't already exist
export const CreateNewUser = inngest.createFunction(
  { id: "create-user" },
  { event: "user.create" },
  async ({ event, step }) => {
    const [user] = event.data;

    const result = await step.run(
      "Check User and Create if Not in DB",
      async () => {
        // Check if the user exists in the database
        const existingUser = await db
          .select()
          .from(USER_TABLE)
          .where(eq(USER_TABLE.email, user?.primaryEmailAddress?.emailAddress));

        if (existingUser.length === 0) {
          // Insert the new user if not found
          const newUser = await db
            .insert(USER_TABLE)
            .values({
              name: user?.fullName,
              email: user?.primaryEmailAddress?.emailAddress,
            })
            .returning({ id: USER_TABLE.id });
          return newUser;
        }
        return existingUser;
      }
    );

    console.log(result);
    return "Success";
  }
);

// Function to generate notes for chapters in a course
export const GenerateNotes = inngest.createFunction(
  { id: "generate-course" },
  { event: "notes.generate" },
  async ({ event, step }) => {
    const { course } = event.data;

    try {
      // Generate notes for each chapter
      const notesResult = await step.run("Generate Chapter Notes", async () => {
        const chapters = course.courseLayout.chapters;

        const chapterPromises = chapters.map(async (chapter, index) => {
          const PROMPT = `Generate a JSON object that represents study notes for a course chapter. The JSON should meet the following requirements:
0. Provided Chapters:
${JSON.stringify(chapter)}


1. Structure:
The JSON must include the following fields:
chapterTitle: The title of the chapter.
chapterSummary: A brief summary of the chapter.
emoji: A relevant emoji to visually represent the chapter.
topics: A list of topics covered in the chapter. Each topic must be an object with:
topicTitle (string): The title of the topic.
content (string): Detailed content for the topic written in Md format, and ready for rendering in a React.js component.

OUTPUT SHOULD BE LIKE : 
{
  "chapterTitle": "WordPress Fundamentals",
  "chapterSummary": "Introduction to WordPress, its architecture, core components, and installation process.",
  "emoji": "🌱",
  "topics": [
    {
      "topicTitle": "What is WordPress?",
      "content": "# What is WordPress? 🤔\n\nWordPress is a free and open-source content management system (CMS) used to build and manage websites and blogs. Its popularity stems from its user-friendly interface, extensive plugin ecosystem, and robust theme customization options. \n\n**Key Features:**\n\n* **Ease of Use:**  Intuitive interface, making it accessible to beginners and experts alike.\n* **Flexibility:**  Highly adaptable to various website needs through themes and plugins.\n* **Extensibility:** Thousands of plugins add functionality, extending core capabilities.\n* **SEO-Friendly:** Built-in features aid search engine optimization.\n* **Large Community:**  Extensive support network provides assistance and resources.\n\n**WordPress Editions:**\n\n* **WordPress.org (Self-hosted):**  You control hosting, offering maximum flexibility and customization. Requires technical setup.\n* **WordPress.com (Hosted):**  WordPress handles hosting, simplifying setup but limiting customization options."
    },
    {
      "topicTitle": "WordPress Architecture",
      "content": "# WordPress Architecture ⚙️\n\nWordPress follows a three-tier architecture:\n\n1. **Presentation Layer (Frontend):**  The user interface; what visitors see.  Managed primarily by themes.\n2. **Application Layer:** The core WordPress engine, processing requests, database interactions, and plugin execution.\n3. **Data Layer:**  The MySQL database storing content, settings, users, and other website data.\n\n**Key Components:**\n\n* **Core Files:** The fundamental WordPress files responsible for core functionality.\n* **Database (MySQL):** Stores all website information.\n* **Themes:** Control the website's visual presentation and layout.\n* **Plugins:** Extend functionality by adding features (e.g., contact forms, e-commerce).\n\n**Simplified Diagram:**\n\n  '\nUser --> Presentation Layer (Theme) --> Application Layer (Core + Plugins) --> Data Layer (Database)\n'"
    },
    {
      "topicTitle": "Installation and Setup",
      "content": "# Installation and Setup 💻\n\n**Prerequisites:**\n\n* **Web Hosting:**  A hosting provider supporting PHP, MySQL, and databases (e.g., Bluehost, SiteGround).\n* **Domain Name:** A registered domain name (e.g., 'yourwebsite.com').\n* **WordPress Files:** Download the latest version from wordpress.org.\n\n**Steps:**\n\n1. **Upload:** Upload the downloaded files to your web hosting account via FTP or your hosting control panel.\n2. **Database Creation:** Create a MySQL database and user account in your hosting control panel.\n3. **Configuration:**  During installation, provide database credentials (name, username, password, hostname).\n4. **Installation:** Follow the on-screen instructions in your browser to complete setup.\n5. **Initial Settings:** Set the site title, admin username and password, and other basic settings.\n\n**Troubleshooting:**\n\n* **Database Errors:** Double-check database credentials.\n* **Permission Issues:** Ensure correct file permissions on your server.\n* **Resource Limits:**  Check server resources (PHP memory limit)."
    },
    {
      "topicTitle": "WordPress Dashboard Overview",
      "content": "# WordPress Dashboard Overview 📊\n\nThe WordPress dashboard is the central administration interface. Key sections include:\n\n* **Dashboard:**  Displays recent activity, quick drafts, and notifications.\n* **Posts:** Manage blog posts (create, edit, publish, schedule).\n* **Pages:** Manage static pages (e.g., About Us, Contact).\n* **Media:** Upload and manage images, videos, and other media files.\n* **Appearance:** Customize themes and appearance.\n* **Plugins:** Install, activate, and manage plugins.\n* **Users:** Manage user accounts and roles.\n* **Settings:** Configure general website settings.\n\n**Navigation:**  Become familiar with the dashboard's layout for efficient website management. 📌"
    },
    {
      "topicTitle": "Understanding WordPress Themes and Plugins",
      "content": "# WordPress Themes and Plugins ✨\n\n**Themes:** Themes control the website's visual design and layout. \n\n* **Theme Selection:** Choose themes based on website purpose and design preferences.\n* **Customization:** Many themes offer options to customize colors, fonts, and layouts.\n* **Child Themes:**  Creating a child theme protects customizations when updating the parent theme.\n\n**Plugins:** Plugins extend WordPress functionality.\n\n* **Plugin Selection:** Choose reputable plugins from WordPress.org.\n* **Compatibility:**  Be mindful of plugin conflicts – test thoroughly before activating multiple plugins.\n* **Updates:** Regularly update plugins for security and bug fixes.\n\n**Examples:**\n\n* **Popular Themes:**  Astra, OceanWP, GeneratePress\n* **Popular Plugins:** Yoast SEO, WooCommerce, Akismet"
    }
  ]
}


2. Content Formatting:
Give me in .md format


**IMPORTANT**
There should be an emoji
Give me in .md format



 7. **Additional Notes:**  
   - **IMPORTANT** There should be an emoji
   - Every Content should be in detail and explained properly
   - Each 'content' field should use simple and concise language suitable for study notes.  
   - Ensure that topics include clear definitions, key points, and, where appropriate, examples or sample code.  
   - All generated content should be focused on clarity and exam preparation, with minimal redundancy.  

 8. **Avoid Common Errors:**  
   - Double-check for mismatched brackets, missing fields, or improperly formatted strings.  `;

          // Generate notes using AI model
          const result = await generateNotesAiModel.sendMessage(PROMPT);
          const aiResp = await result.response.text();

          // Insert the generated notes into the database
          await db.insert(CHAPTER_NOTES_TABLE).values({
            chapterId: index,
            courseId: course.courseId,
            notes: aiResp,
          });
        });

        // Wait for all chapter notes to be processed
        await Promise.all(chapterPromises);
        return "Chapter Notes Generated";
      });

      // Update course status to "Ready"
      const updateCourseStatusResult = await step.run(
        "Update Course Status to Ready",
        async () => {
          await db
            .update(STUDY_MATERIAL_TABLE)
            .set({ status: "Ready" })
            .where(eq(STUDY_MATERIAL_TABLE.courseId, course.courseId));
          return "Course Status Updated to Ready";
        }
      );

      // Return the combined results
      return { notesResult, updateCourseStatusResult };
    } catch (error) {
      // Log and rethrow any errors
      console.error("Error during GenerateNotes function execution:", error);
      throw new Error("An error occurred while generating course notes");
    }
  }
);

export const GenerateStudyTypeContent = inngest.createFunction(
  { id: "Generate Study Type Content" },
  { event: "studyType.content" },
  async ({ event, step }) => {
    const { studyType, prompt, courseId, recordId } = event.data;
    const AIResult = await step.run(
      "Generating Flashcard using AI",
      async () => {
        const result =
          studyType == "Flashcard"
            ? await GenerateStudyTypeContentAiModel.sendMessage(prompt)
            : studyType == "Quiz"
            ? await GenerateQuizAiModel.sendMessage(prompt)
            : await GenerateQaAiModel.sendMessage(prompt);
        const AIResult = JSON.parse(result.response.text());
        return AIResult;
      }
    );
    const DbResult = await step.run("Save Result to DB", async () => {
      const result = await db
        .update(STUDY_TYPE_CONTENT_TABLE)
        .set({
          content: AIResult,
          status: "Ready",
        })
        .where(eq(STUDY_TYPE_CONTENT_TABLE.id, recordId));
      return "Data Inserted";
    });
  }
);
