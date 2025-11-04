import { GoogleGenAI } from "@google/genai";
import { ClusterResult } from '../types';

const API_KEY = import.meta.env.VITE_GEMINI_API_KEY;

export const analyzeClusteringResults = async (results: ClusterResult[], datasetName: string): Promise<string> => {
  if (!API_KEY) {
    console.error("API Key not found:", import.meta.env);
    return "Error: VITE_GEMINI_API_KEY environment variable not set. Please configure it to use Gemini analysis.";
  }
  
  const genAI = new GoogleGenAI({ apiKey: API_KEY });
  const datasetDescriptions: Record<string, string> = {
    'Moons': 'Two interlocking, non-convex crescent shapes. A classic test for algorithms that can handle non-linear structures.',
    'Circles': 'Two concentric circles. A key test for algorithms that are not based on linear separability, where Spectral Clustering should excel.',
    'Blobs': 'Several distinct, globular (Gaussian) clusters. Well-suited for centroid-based algorithms like K-Means.',
  };

  const datasetDescription = datasetDescriptions[datasetName] || 'A custom user-provided dataset.';


  const prompt = `
    You are an expert data scientist specializing in clustering algorithms.
    Your task is to analyze and compare the performance of four clustering algorithms on a given dataset.

    Dataset Name: ${datasetName}
    Dataset Characteristics: ${datasetDescription}

    Here are the results and metrics for each algorithm:
    ${results.map(r => `
    ---
    Algorithm: ${r.algorithm}
    Parameters: ${JSON.stringify(r.params)}
    Metrics:
      - Silhouette Score: ${r.metrics.silhouette} (higher is better, range -1 to 1)
      - Calinski-Harabasz Index: ${r.metrics.calinskiHarabasz} (higher is better)
      - Davies-Bouldin Score: ${r.metrics.daviesBouldin} (lower is better, closer to 0)
    `).join('\n')}
    ---

    Based on the provided metrics AND the known characteristics of the dataset, please provide a comprehensive analysis in Markdown format. Your analysis should include:

    1.  **Overall Summary:** A brief conclusion about which algorithm appears to be the most suitable for this dataset and why.
    2.  **Algorithm Comparison:**
        *   Discuss the performance of Spectral Clustering compared to the others. Relate its performance directly to the geometric properties of the '${datasetName}' dataset. For the 'Circles' dataset, specifically highlight how it succeeds where others fail.
        *   Compare K-Means, Agglomerative Clustering, and DBSCAN, referencing their metrics and how the dataset shape impacts their performance.
    3.  **Metric Interpretation:** Briefly explain what the combination of Silhouette, Calinski-Harabasz, and Davies-Bouldin scores indicates about the quality of the clusters (e.g., density, separation).
    4.  **Recommendation:** Conclude with a clear recommendation for the best performing algorithm for the '${datasetName}' dataset, justifying your choice based on both the metrics and the data's geometry.
    5. **working of spectral clustering for the particular dataset.
  `;

  try {
    // Use a model that your API key supports. The "models/*" names are returned by
    // the REST ListModels endpoint. Based on the key in your .env, use a supported
    // model id below.
    // The client package's API surface varies by version. Use a flexible approach
    // that tries the high-level client method first and falls back to common shapes.
    const clientAny: any = genAI;

    // Try models.generateContent() (newer clients)
    let rawResult: any = null;
    if (clientAny.models && typeof clientAny.models.generateContent === 'function') {
      rawResult = await clientAny.models.generateContent({ model: "models/gemini-2.5-flash", contents: prompt });
    } else if (typeof clientAny.getGenerativeModel === 'function') {
      // older style: get a model handle then call generateContent
      const modelHandle = clientAny.getGenerativeModel({ model: "models/gemini-2.5-flash" });
      rawResult = await modelHandle.generateContent(prompt);
    } else {
      throw new Error('Unsupported Google GenAI client usage: cannot find generation method on client');
    }

    // Normalize possible response shapes to a string. The newer Gemini responses
    // often contain `candidates` -> `content` -> `parts` -> `text` (or `outputs`).
    if (rawResult == null) return 'No response from model';
    try {
      // If the SDK returned a helper with .text() method
      if (typeof rawResult.text === 'function') {
        return await rawResult.text();
      }
      if (rawResult.response && typeof rawResult.response.text === 'function') {
        return await rawResult.response.text();
      }

      // Newer REST-like shapes: { candidates: [ { content: { parts: [ { text } ] } } ] }
      if (Array.isArray(rawResult.candidates) && rawResult.candidates.length > 0) {
        const candidate = rawResult.candidates[0];
        // content.parts is common
        if (candidate.content && Array.isArray(candidate.content.parts)) {
          const parts = candidate.content.parts.map((p: any) => p.text || p).filter(Boolean);
          if (parts.length) return parts.join('\n\n');
        }
        // fallback to any text field on candidate
        if (typeof candidate === 'string') return candidate;
        if (typeof candidate.content === 'string') return candidate.content;
      }

      // Alternate shape: outputs: [{ content: [{ text: '...' }] }]
      if (Array.isArray(rawResult.outputs) && rawResult.outputs.length > 0) {
        const out = rawResult.outputs[0];
        if (out.content && Array.isArray(out.content)) {
          const part = out.content[0];
          if (part && part.text) return part.text;
        }
        if (typeof out === 'string') return out;
      }

      // Simple properties
      if (typeof rawResult.outputText === 'string') return rawResult.outputText;

      // As a last resort, pretty-print the JSON but strip large sdkHttpResponse details
      const copy = { ...rawResult } as any;
      if (copy.sdkHttpResponse) delete copy.sdkHttpResponse;
      return JSON.stringify(copy, null, 2);
    } catch (e) {
      return `Error parsing model response: ${(e as Error).message}`;
    }
  } catch (error) {
    console.error("Error calling Gemini API:", error);
    if (error instanceof Error) {
        return `An error occurred while analyzing the results with Gemini: ${error.message}`;
    }
    return "An unknown error occurred while analyzing the results with Gemini.";
  }
};