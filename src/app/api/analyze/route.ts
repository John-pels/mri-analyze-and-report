import { type NextRequest, NextResponse } from "next/server";
import { analyzeWithCNN } from "@/lib/cnn-model";
import { GoogleGenerativeAI } from "@google/generative-ai";

export async function POST(request: NextRequest) {
  try {
    // Get the form data from the request
    const formData = await request.formData();
    const imageFile = formData.get("image") as File | null;
    const prompt = formData.get("prompt") as string | null;
    const imageType = (formData.get("imageType") as string) || "mri";
    // const apiKey = process.env.OPENAI_API_KEY!;
    const geminiKey = process.env.GEMINI_API_KEY!;

    console.log("api key", geminiKey);

    // Validate inputs
    if (!imageFile) {
      return NextResponse.json(
        { error: "No image file provided" },
        { status: 400 }
      );
    }

    // Check file size (3MB limit)
    const MAX_FILE_SIZE = 3 * 1024 * 1024; // 3MB in bytes
    if (imageFile.size > MAX_FILE_SIZE) {
      return NextResponse.json(
        { error: "File too large (max 3MB)" },
        { status: 400 }
      );
    }

    // Convert the image file to an array buffer
    const imageBuffer = await imageFile.arrayBuffer();

    // Process the image with the CNN model
    const cnnResults = await analyzeWithCNN(imageBuffer, imageType);

    // Prepare the prompt for the LLM
    let analysisPrompt = `
You are a medical imaging expert specializing in ${imageType.toUpperCase()} analysis. 
Based on the following CNN model analysis of a ${imageType.toUpperCase()} image, provide a detailed interpretation:

CNN Analysis Results:
${JSON.stringify(cnnResults, null, 2)}

Additional Context:
${
  prompt ||
  `Provide a general analysis of this ${imageType.toUpperCase()} image.`
}

Provide a comprehensive analysis that would be helpful for medical professionals. 
Include potential findings, areas of interest, and any recommendations for further investigation if applicable.
`;

    // Add brain tumor-specific instructions for MRI images
    if (imageType === "mri") {
      analysisPrompt += `
If there are any indications of brain tumors in the analysis, please provide:
1. Potential tumor type based on imaging characteristics
2. Location and size assessment
3. Possible effects on surrounding structures
4. Differential diagnosis considerations
5. Recommended follow-up imaging or procedures
`;
    }

    // // Use the LLM to generate a detailed analysis
    // const { text } = await generateText({
    //   model: openai("gpt-3.5-turbo-instruct"),
    //   prompt: analysisPrompt,
    // });

    // Initialize Gemini
    const genAI = new GoogleGenerativeAI(geminiKey);
    const model = genAI.getGenerativeModel({ model: "gemini-1.5-pro" }); // Specify the model

    // Generate content with Gemini
    const result = await model.generateContent({
      contents: [{ role: "user", parts: [{ text: analysisPrompt }] }],
    });
    const text = result.response.text();

    // Return the analysis results
    return NextResponse.json({
      analysis: text,
      regions: cnnResults.regionsOfInterest,
    });
  } catch (error) {
    console.error("Error processing medical image analysis:", error);
    return NextResponse.json(
      { error: "Failed to analyze the medical image" },
      { status: 500 }
    );
  }
}
