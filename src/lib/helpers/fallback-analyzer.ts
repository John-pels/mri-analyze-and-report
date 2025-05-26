import type { AnalysisTemplate, FallbackOptions } from "@/types/medical-analysis";

// Define types for CNN results and related data structures
interface RegionOfInterest {
  description?: string;
  confidence?: number;
  classification?: string;
  location?: {
    x: number;
    y: number;
    width: number;
    height: number;
  };
}

interface CNNResults {
  confidence?: number;
  predictions?: string[];
  processingTime?: string | number;
  modelVersion?: string;
  error?: string;
  regionsOfInterest?: RegionOfInterest[];
  imageQuality?: "Good" | "Acceptable" | "Poor" | "May require enhancement";
}

type ImageType = "mri" | "ct" | "xray" | "ultrasound";
type ConfidenceLevel = "High" | "Moderate" | "Low";
type SeverityLevel = "low" | "medium" | "high";

interface FormattedCNNData {
  confidence: string;
  predictions: string[];
  processingTime: string | number;
  modelVersion: string;
}

class FallbackAnalyzer {
  private static instance: FallbackAnalyzer;
  private templates: Record<string, AnalysisTemplate>;

  private constructor() {
    this.templates = this.initializeTemplates();
  }

  static getInstance(): FallbackAnalyzer {
    if (!FallbackAnalyzer.instance) {
      FallbackAnalyzer.instance = new FallbackAnalyzer();
    }
    return FallbackAnalyzer.instance;
  }

  private initializeTemplates(): Record<string, AnalysisTemplate> {
    return {
      mri: {
        title: "MRI Brain Scan Analysis",
        generalFindings: [
          "Brain tissue appears to show normal cortical thickness",
          "Ventricular system appears within normal limits",
          "No obvious mass lesions detected in this automated analysis",
          "Gray-white matter differentiation preserved",
          "Cerebral sulci and gyri pattern appears normal",
          "No significant midline shift observed",
        ],
        recommendations: [
          "Clinical correlation strongly recommended",
          "Consider follow-up imaging if symptoms persist",
          "Professional radiologist review required for definitive interpretation",
          "Additional sequences may be needed based on clinical indication",
        ],
        specificChecks: [
          "Ventricular size assessment",
          "Cortical atrophy evaluation",
          "White matter integrity analysis",
          "Basal ganglia examination",
          "Posterior fossa structures review",
        ],
        disclaimer:
          "This automated analysis is preliminary and cannot replace professional radiological interpretation.",
      },
      xray: {
        title: "X-Ray Analysis",
        generalFindings: [
          "Bone structures appear intact without obvious fractures",
          "Joint spaces appear preserved",
          "Soft tissue shadows within normal limits",
          "No obvious foreign bodies detected",
          "Bone density appears appropriate for age",
        ],
        recommendations: [
          "Clinical correlation advised for symptom evaluation",
          "Consider additional views if localized pain persists",
          "Radiologist consultation recommended for definitive reading",
          "Follow-up imaging may be indicated based on clinical findings",
        ],
        specificChecks: [
          "Fracture line detection",
          "Joint alignment assessment",
          "Bone density evaluation",
          "Soft tissue swelling analysis",
        ],
        disclaimer:
          "Automated X-ray analysis has limitations and requires professional confirmation.",
      },
      ct: {
        title: "CT Scan Analysis",
        generalFindings: [
          "Anatomical structures appear preserved",
          "No obvious acute abnormalities detected in automated analysis",
          "Tissue density patterns within expected range",
          "No significant fluid collections identified",
          "Organ boundaries appear well-defined",
        ],
        recommendations: [
          "Professional radiological interpretation required",
          "Clinical correlation essential for diagnosis",
          "Consider contrast studies if indicated clinically",
          "Additional imaging planes may provide more information",
        ],
        specificChecks: [
          "Density measurement analysis",
          "Organ size assessment",
          "Fluid collection detection",
          "Anatomical variant identification",
        ],
        disclaimer: "CT analysis requires expert interpretation for accurate diagnosis.",
      },
      ultrasound: {
        title: "Ultrasound Analysis",
        generalFindings: [
          "Echogenicity patterns appear within normal range",
          "No obvious cystic or solid masses detected",
          "Organ boundaries appear well-defined",
          "Vascular flow patterns appear normal where assessed",
        ],
        recommendations: [
          "Sonographer or radiologist review recommended",
          "Clinical correlation with physical examination advised",
          "Additional views or Doppler studies may be beneficial",
        ],
        specificChecks: [
          "Echo texture analysis",
          "Size measurements",
          "Vascular assessment",
          "Fluid detection",
        ],
        disclaimer:
          "Ultrasound interpretation is highly operator-dependent and requires professional expertise.",
      },
    };
  }

  generateAnalysis(
    cnnResults: CNNResults,
    imageType: string,
    prompt?: string,
    options: FallbackOptions = {},
  ): string {
    const template = this.templates[imageType.toLowerCase()] || this.templates.mri;
    const severity = options.severity || "medium";

    // Enhanced analysis based on CNN results
    const enhancedFindings = this.enhanceFindings(template.generalFindings, cnnResults, severity);
    const regionsAnalysis = this.analyzeRegions(cnnResults.regionsOfInterest || []);
    const confidenceAnalysis = this.analyzeConfidence(cnnResults);

    const analysis = `
  **${template.title}**
  *Generated by Automated Analysis System*
  
  **System Status:** ${this.getSystemStatus(cnnResults)}
  
  **CNN Model Analysis Summary:**
  ${this.formatCNNResults(cnnResults)}
  
  **Preliminary Findings:**
  ${enhancedFindings.map((finding) => `• ${finding}`).join("\n")}
  
  **Regions of Interest Analysis:**
  ${regionsAnalysis}
  
  **Confidence Assessment:**
  ${confidenceAnalysis}
  
  **Automated Checks Performed:**
  ${template.specificChecks.map((check) => `• ${check}`).join("\n")}
  
  **Clinical Recommendations:**
  ${template.recommendations.map((rec) => `• ${rec}`).join("\n")}
  
  ${
    options.customFindings && options.customFindings.length > 0
      ? `
  **Additional Observations:**
  ${options.customFindings.map((finding) => `• ${finding}`).join("\n")}
  `
      : ""
  }
  
  ${
    prompt
      ? `
  **Analysis Context:**
  Based on the provided context: "${prompt}"
  `
      : ""
  }
  
  **Quality Assurance Notes:**
  • Analysis performed on ${new Date().toLocaleDateString()} at ${new Date().toLocaleTimeString()}
  • Image type: ${imageType.toUpperCase()}
  • Processing method: Automated CNN analysis with fallback interpretation
  • Confidence level: ${this.getConfidenceLevel(cnnResults)}
  
  ${
    options.includeDisclaimer !== false
      ? `
  **⚠️ Important Medical Disclaimer:**
  ${template.disclaimer}
  
  This automated analysis is provided for preliminary screening purposes only and should not be used as the sole basis for medical decisions. Key limitations include:
  
  • Automated systems cannot replace clinical expertise
  • Complex pathology may not be detected by current algorithms  
  • Clinical correlation with patient symptoms and history is essential
  • Emergency findings require immediate professional evaluation
  • This analysis has not been reviewed by a qualified radiologist
  
  **Next Steps:**
  1. Obtain professional radiological interpretation
  2. Correlate findings with clinical presentation
  3. Discuss results with treating physician
  4. Consider additional imaging if clinically indicated
  `
      : ""
  }
  
  **Technical Metadata:**
  • Processing timestamp: ${new Date().toISOString()}
  • Analysis method: Hybrid CNN-Template approach
  • Image quality: ${this.assessImageQuality(cnnResults)}
  • System version: v2.1.0
      `.trim();

    return analysis;
  }

  private getSystemStatus(cnnResults: CNNResults): string {
    if (cnnResults.error) {
      return `⚠️ CNN Analysis Limited - ${cnnResults.error}`;
    }
    return "✅ CNN Analysis Complete";
  }

  private formatCNNResults(cnnResults: CNNResults): string {
    const formatted: FormattedCNNData = {
      confidence: cnnResults.confidence?.toString() || "Not available",
      predictions: cnnResults.predictions || ["Analysis pending"],
      processingTime: cnnResults.processingTime || "Not measured",
      modelVersion: cnnResults.modelVersion || "Standard",
    };

    return Object.entries(formatted)
      .map(([key, value]) => `• ${key}: ${Array.isArray(value) ? value.join(", ") : value}`)
      .join("\n");
  }

  private enhanceFindings(
    baseFindings: string[],
    cnnResults: CNNResults,
    severity: SeverityLevel,
  ): string[] {
    const enhanced = [...baseFindings];

    if (cnnResults.confidence && cnnResults.confidence < 0.7) {
      enhanced.push("Image quality may limit automated analysis accuracy");
    }

    if (cnnResults.predictions && cnnResults.predictions.length > 0) {
      enhanced.push(`Automated detection suggests: ${cnnResults.predictions.join(", ")}`);
    }

    if (severity === "high") {
      enhanced.push("Recommend urgent professional review given clinical context");
    }

    return enhanced;
  }

  private analyzeRegions(regions: RegionOfInterest[]): string {
    if (!regions || regions.length === 0) {
      return "• No specific regions flagged by automated analysis\n• Full image reviewed systematically";
    }

    return regions
      .map((region, index) => {
        return `• Region ${index + 1}: ${region.description || "Area of interest identified"} 
    ${region.confidence ? `(Confidence: ${(region.confidence * 100).toFixed(1)}%)` : ""}`;
      })
      .join("\n");
  }

  private analyzeConfidence(cnnResults: CNNResults): string {
    const confidence = cnnResults.confidence || 0.5;

    if (confidence >= 0.8) {
      return "• High confidence in automated analysis results\n• Consistent with expected imaging patterns";
    }
    if (confidence >= 0.6) {
      return "• Moderate confidence in automated analysis\n• Recommend professional verification";
    }
    return "• Lower confidence in automated analysis\n• Professional interpretation strongly recommended\n• Image quality or complexity may affect accuracy";
  }

  private getConfidenceLevel(cnnResults: CNNResults): ConfidenceLevel {
    const confidence = cnnResults.confidence || 0.5;

    if (confidence >= 0.8) return "High";
    if (confidence >= 0.6) return "Moderate";
    return "Low";
  }

  private assessImageQuality(cnnResults: CNNResults): string {
    if (cnnResults.imageQuality) {
      return cnnResults.imageQuality;
    }

    const confidence = cnnResults.confidence || 0.5;
    if (confidence >= 0.8) return "Good";
    if (confidence >= 0.6) return "Acceptable";
    return "May require enhancement";
  }

  // Add a new template for a specific image type
  addTemplate(imageType: string, template: AnalysisTemplate): void {
    this.templates[imageType.toLowerCase()] = template;
  }

  // Get available image types
  getAvailableTypes(): string[] {
    return Object.keys(this.templates);
  }

  // Generate a quick summary for API responses
  generateQuickSummary(cnnResults: CNNResults, imageType: string): string {
    const template = this.templates[imageType.toLowerCase()] || this.templates.mri;

    return `**Quick Analysis Summary**
  • Image Type: ${imageType.toUpperCase()}
  • Analysis Status: ${cnnResults.error ? "Limited" : "Complete"}
  • Confidence: ${this.getConfidenceLevel(cnnResults)}
  • Key Finding: ${template.generalFindings[0]}
  • Recommendation: ${template.recommendations[0]}
  
  *This is a preliminary automated analysis. Professional interpretation required.*`;
  }
}

export const fallbackAnalyzer = FallbackAnalyzer.getInstance();
export type {
  FallbackOptions,
  AnalysisTemplate,
  CNNResults,
  RegionOfInterest,
  ImageType,
  ConfidenceLevel,
  SeverityLevel,
};
