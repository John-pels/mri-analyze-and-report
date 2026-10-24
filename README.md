# Medical Image Analysis Assistant

A web application that classifies brain MRI slices **in the browser** with a convolutional neural network (CNN) and drafts a preliminary text report with a large language model (LLM). The image never leaves the device: only the CNN's class probabilities and the user's question are sent to the server.

> **Research prototype — not a medical device.** Outputs are screening aids for research and teaching. They are not a diagnosis and must be reviewed by a qualified clinician.

## What it does

1. You upload a brain MRI slice (JPEG, PNG or GIF, up to 3 MB).
2. A TensorFlow.js CNN in the browser classifies it as **no tumour, glioma, meningioma or pituitary tumour**.
3. For tumour predictions, **Grad-CAM** draws a box around the region that most influenced the prediction.
4. The browser sends the class probabilities, the Grad-CAM box and your prompt (as JSON, no pixels) to `/api/analyze`.
5. The server asks **Google Gemini** to draft a report. If Gemini fails it tries **OpenAI**; if both fail, or no key is set, a **template generator** writes a report that only restates the CNN output.

Only brain MRI has a trained model. CT, X-ray and ultrasound appear in the menu but are disabled.

## Model performance

The four-class model was trained on the public [Brain Tumor Classification (MRI)](https://github.com/SartajBhuvaji/Brain-Tumor-Classification-DataSet) dataset (MIT licence) after removing duplicate images, and tested once on 354 held-out images.

| Measure | Result |
| --- | --- |
| Accuracy | 78.8% (95% CI 74.3–82.8) |
| Cohen's κ | 0.71 |
| Sensitivity: no tumour / glioma / meningioma / pituitary | 88.4% / 80.7% / 62.1% / 90.4% |
| Tumour vs no tumour: sensitivity / specificity / AUC | 95.5% / 86.0% / 0.963 |
| Browser vs Python predictions | Identical class on all 354 test images |

Grad-CAM works on an 8 × 8 feature map, so its boxes are coarse and often highlight the skull or image edge rather than the lesion. Treat them as a rough cue only. Full results are in [`research/results/`](research/results/).

## Requirements

- [Bun](https://bun.sh) 1.1 or later (or Node.js 18.18+ with npm)
- A modern browser (Chrome, Edge, Firefox or Safari)
- Optional: a Google Gemini and/or OpenAI API key for LLM reports

## Run it locally

```bash
git clone https://github.com/John-pels/mri-analyze-and-report.git
cd mri-analyze-and-report

bun install
cp .env.example .env.local   # then add your keys (optional)
bun dev
```

Open http://localhost:3000, choose **MRI Scan**, upload an image and click **Analyze Medical Image**.

Without API keys the app still works: the CNN and Grad-CAM run as normal and the report comes from the template generator.

### Production build

```bash
bun run build
bun start
```

### Using npm instead of Bun

```bash
npm install
npm run dev
```

### Environment variables

| Variable | Required | Purpose |
| --- | --- | --- |
| `GEMINI_API_KEY` | No | Google Gemini key; used first for reports (`gemini-1.5-flash`) |
| `OPENAI_API_KEY` | No | OpenAI key; used if Gemini fails or is not set (`gpt-3.5-turbo`) |

Keys stay on the server and are never sent to the browser. See `.env.example`.

### Lint and type-check

```bash
bun run check-types         # Biome lint + format (writes fixes)
npx tsc --noEmit            # TypeScript
```

## Project structure

```
src/
├── app/
│   ├── api/analyze/route.ts   # Report drafting: rate limit, cache, Gemini → OpenAI → template
│   ├── layout.tsx
│   └── page.tsx
├── components/
│   ├── screen/index.tsx       # Main screen
│   ├── image-viewer.tsx       # Zoom, rotate, pan, Grad-CAM box overlay
│   └── ui/                    # shadcn/ui components
├── hooks/
│   ├── use-image-analysis.ts  # Upload and analysis state
│   └── use-dropzone-handler.ts
├── lib/
│   ├── cnn-model.ts           # In-browser CNN, preprocessing, Grad-CAM
│   ├── analyze-image.ts       # Runs the CNN, then calls the API with JSON only
│   └── helpers/               # Cache, rate limiter, usage monitor, template report
└── types/
public/models/mri/             # Trained TensorFlow.js model (model.json + weights.bin, 3.7 MB)
research/                      # Training, evaluation and export code; results
```

## The CNN

| Layer | Detail |
| --- | --- |
| Input | 64 × 64 × 1 grayscale, scaled to [0, 1] |
| 4 convolutional blocks | 3 × 3 conv (32, 64, 128, 256 filters) + ReLU + 2 × 2 max pool |
| 2 dense layers | 128 units + ReLU + dropout 0.5 |
| Output | 4-unit softmax |

929,284 parameters. Preprocessing in the browser matches training exactly.

## Retraining the model

The `research/` folder holds the Python pipeline used for the model and the accompanying papers: deduplication, the 108-configuration grid search, test-set evaluation with confidence intervals, and export to TensorFlow.js. See [`research/README.md`](research/README.md). After retraining, copy `research/runs/tfjs_mri/model.json` and `weights.bin` into `public/models/mri/`.

## Privacy and security

- Images are processed in the browser and are not uploaded.
- The server receives only class probabilities, the Grad-CAM box, the model version and the prompt.
- That text is sent to Google or OpenAI when keys are set, under their API terms. For real patient data, use a data-processing agreement or a self-hosted model.
- The server applies per-IP and global rate limits and caches reports by exact match only.

## Limitations

- Brain MRI only; single 2D slices at 64 × 64 pixels.
- Trained and tested on one public dataset; not validated on clinical scanners.
- Weak at separating meningioma from glioma.
- Grad-CAM localisation is coarse and unvalidated.
- LLM report quality has not been evaluated.

## Citation

If you use this software, please cite: Emmanuel JO, Udomboso CG, Sigauke C, Edeki G. *Medical Image Analysis Assistant: an open-source web application for in-browser brain MRI classification with large language model report drafting* (manuscript in preparation).

## License

Released under the [MIT License](LICENSE). The licence covers the code and the trained model in `public/models/mri/`. The training dataset is a separate work by its authors, also under the MIT licence. The software is provided "as is", without warranty, and is not approved for clinical use.

## Acknowledgements

Brain Tumor Classification (MRI) dataset by Sartaj Bhuvaji et al.; TensorFlow.js; Next.js; Radix UI and shadcn/ui; the Associates Programme of the Quantitative Life Sciences Section, Abdus Salam International Centre for Theoretical Physics, Trieste; and the Department of Statistics, University of Ibadan.
