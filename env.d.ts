export {};

declare global {
  namespace NodeJS {
    interface ProcessEnv extends NodeJS.ProcessEnv {
      OPENAI_API_KEY: string;
      GEMINI_API_KEY: string;
    }
  }
}
