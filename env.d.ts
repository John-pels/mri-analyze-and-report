export {};

declare global {
  namespace NodeJS {
    interface ProcessEnv extends NodeJS.ProcessEnv {
      NEXT_PUBLIC_OPENAI_API_KEY: string;
    }
  }
}
