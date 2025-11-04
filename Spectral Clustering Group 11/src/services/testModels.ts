import { GoogleGenAI } from "@google/genai";

const API_KEY = process.env.VITE_GEMINI_API_KEY || (typeof import.meta !== 'undefined' ? (import.meta as any).env?.VITE_GEMINI_API_KEY : undefined);

async function listModels() {
    if (!API_KEY) {
        console.error("API key not found!");
        return;
    }

    const genAI = new GoogleGenAI({ apiKey: API_KEY as string });
    console.log("Created GoogleGenAI client (test).\nNote: this script is intended for Node, and may not run in the browser.");
}

listModels();