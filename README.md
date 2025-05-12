# Medical Image Analysis Assistant

A sophisticated web application that uses Convolutional Neural Networks (CNN) and Large Language Models (LLM) to analyze medical images and provide detailed insights.

## Features

- **Multi-modal Medical Image Analysis**: Support for MRI, CT, X-Ray, and Ultrasound images
- **Advanced CNN Processing**: Utilizes TensorFlow.js to process images directly in the browser
- **LLM-Powered Analysis**: Leverages OpenAI's GPT-4o model to generate detailed medical insights
- **Interactive Image Viewer**: Zoom, rotate, and pan functionality with region of interest highlighting
- **Customizable Analysis Types**: Choose between general analysis, detailed reports, or medical insights
- **Custom Prompts**: Tailor the analysis with custom instructions
- **Responsive Design**: Works seamlessly across desktop and mobile devices
- **File Validation**: Ensures images are under 3MB and in supported formats

## Technology Stack

- **Frontend**: Next.js 14 with React 18
- **UI Components**: Custom components built with Radix UI and styled with Tailwind CSS
- **AI Integration**: Vercel AI SDK with OpenAI integration
- **Image Processing**: TensorFlow.js for CNN model implementation
- **State Management**: React Hooks with custom hook abstractions
- **Styling**: Tailwind CSS with shadcn/ui component system
- **Code Quality**: Biome for linting and formatting

## Architecture

The application follows a clean architecture with separation of concerns:

```
├── app/                  # Next.js App Router
│   ├── api/              # API routes for server-side processing
│   ├── layout.tsx        # Root layout
│   └── page.tsx          # Main application page
├── components/           # UI components
│   ├── image-viewer.tsx  # Interactive image viewer component
│   └── ui/               # Base UI components (shadcn/ui)
├── hooks/                # Custom React hooks
│   ├── use-dropzone-handler.ts  # File upload handling
│   ├── use-image-analysis.ts    # Image analysis logic
│   ├── use-toast.ts             # Toast notification system
│   └── use-mobile.tsx           # Responsive design utilities
├── lib/                  # Utility functions and core logic
│   ├── analyze-image.ts  # Image analysis pipeline
│   ├── cnn-model.ts      # CNN model implementation
│   └── utils.ts          # General utility functions
└── public/               # Static assets
```

## Getting Started

### Prerequisites

- Node.js 18.0 or later
- npm, yarn, or pnpm

### Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/yourusername/medical-image-analysis.git
   cd medical-image-analysis
   ```

2. Install dependencies:

   ```bash
   npm install
   # or
   yarn
   # or
   pnpm install
   ```

3. Set up environment variables:
   Create a `.env.local` file in the root directory with the following variables:

```bash
OPENAI_API_KEY=your_openai_api_key
```

4. Start the development server:

```bash
npm run dev
# or
yarn dev
# or
pnpm dev
```

5. Open [http://localhost:3000](http://localhost:3000) in your browser to see the application.

## Usage

1. **Upload an Image**: Drag and drop a medical image or click to browse files
2. **Select Image Type**: Choose between MRI, CT, X-Ray, or Ultrasound
3. **Choose Analysis Type**: Select general analysis, detailed report, or medical insights
4. **Customize Prompt (Optional)**: Modify the default prompt or add specific instructions
5. **Analyze**: Click the "Analyze Medical Image" button to process the image
6. **View Results**: Examine the analysis results and detected regions of interest
7. **Interact with Image**: Use the image viewer controls to zoom, rotate, and pan

## CNN Model Architecture

The application includes a modular CNN architecture that adapts to different types of medical images:

- **MRI Scans**: Optimized for grayscale 3D volumetric data with focus on soft tissue contrast
- **CT Scans**: Configured for density-based imaging with attention to bone and tissue boundaries
- **X-Ray**: Designed for 2D projection radiography with emphasis on bone and dense structures
- **Ultrasound**: Tailored for acoustic impedance imaging with speckle pattern analysis

Each model variant includes:

- Convolutional layers for feature extraction
- Max pooling for dimensionality reduction
- Dropout layers for regularization
- Dense layers for classification

## LLM Integration

The application uses the AI SDK to integrate with OpenAI's GPT-4o model:

```typescript
const { text } = await generateText({
  model: openai("gpt-4o"),
  prompt: analysisPrompt,
});
```

The LLM receives structured data from the CNN analysis and generates a comprehensive report that includes:

- General observations
- Potential abnormalities
- Anatomical descriptions
- Recommendations for further investigation

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- TensorFlow.js team for the excellent machine learning library
- OpenAI for providing the GPT models
- Vercel for the AI SDK and hosting platform
- shadcn/ui for the beautiful component system
