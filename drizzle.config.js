import { defineConfig } from "drizzle-kit";

export default defineConfig({
  dialect: "postgresql",
  schema: "./configs/schema.js",
  dbCredentials: {
    url: "postgresql://ai-lms_owner:ib2y5Dptxnqe@ep-fragrant-star-a5jhneun.us-east-2.aws.neon.tech/ai-lms?sslmode=require",
  },
});
