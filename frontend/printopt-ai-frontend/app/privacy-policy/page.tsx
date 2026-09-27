import Link from 'next/link';
import { ArrowLeft, ShieldCheck } from 'lucide-react';

export const metadata = {
  title: 'Privacy Policy — PrintOpt AI',
  description:
    'Privacy Policy for PrintOpt AI, an academic research platform for additive manufacturing process optimization.',
};

const sections = [
  {
    title: '1. Introduction',
    content: [
      'PrintOpt AI is an academic research and engineering platform developed for AI-driven process optimization in Laser Powder Bed Fusion (LPBF) of Ti-6Al-4V.',
      'This Privacy Policy explains how information may be handled when you interact with the PrintOpt AI platform. The platform is designed primarily for research, experimentation, data analysis, prediction, and process optimization.',
    ],
  },
  {
    title: '2. Project Information',
    content: [
      'PrintOpt AI is developed as an academic engineering project at the University of the Punjab, Lahore.',
      'Project Team: Hardik Sonu and Samreen Utta Ur Rehman.',
      'Supervisor: Dr. Ing. Waseem Amin.',
    ],
  },
  {
    title: '3. Information We Collect',
    content: [
      'The current PrintOpt AI application does not require user registration, account creation, passwords, profile photographs, or a personal user account.',
      'Process parameters entered into prediction or optimization interfaces may be transmitted to the application backend in order to perform the requested machine-learning computation.',
      'The platform should not be used to submit passwords, payment information, government identification numbers, or other highly sensitive personal information.',
    ],
  },
  {
    title: '4. Research Dataset',
    content: [
      'PrintOpt AI uses experimental and literature-derived additive manufacturing data related to LPBF processing of Ti-6Al-4V.',
      'The research dataset contains process parameters and material-property measurements used for analysis, model development, prediction, and process optimization.',
      'Dataset records are presented for research and engineering purposes and should be interpreted within the limitations of the underlying experimental data.',
    ],
  },
  {
    title: '5. Machine Learning Predictions',
    content: [
      'The platform may process submitted process parameters through machine-learning models to estimate material properties including Ultimate Tensile Strength (UTS), Yield Strength (YS), and Elongation.',
      'Machine-learning predictions are computational estimates and are not a substitute for physical experimentation, laboratory validation, engineering qualification, or manufacturing certification.',
      'Prediction results should therefore be independently evaluated before being used for experimental, industrial, or safety-critical decisions.',
    ],
  },
  {
    title: '6. Process Optimization',
    content: [
      'The optimization module searches within specified process-parameter constraints to identify candidate parameter combinations based on the selected optimization objective.',
      'Recommended parameters are computational recommendations generated from the available dataset and model. They should be experimentally validated before practical manufacturing use.',
    ],
  },
  {
    title: '7. Data Storage and Security',
    content: [
      'PrintOpt AI is designed as a research platform and does not currently provide user accounts or a personal profile system.',
      'Any information transmitted to a backend service is processed according to the configuration and deployment environment of the application.',
      'Users should avoid submitting confidential, proprietary, or personally identifying information through fields that are intended for process parameters or research inputs.',
    ],
  },
  {
    title: '8. Third-Party Services and External Links',
    content: [
      'The platform may contain links to external websites, research publications, repositories, or other resources. These external services operate independently and may have their own privacy policies and terms.',
      'PrintOpt AI does not claim responsibility for the privacy practices or content of external websites.',
    ],
  },
  {
    title: '9. Academic and Research Use',
    content: [
      'PrintOpt AI is developed in an academic and research context. The platform, models, datasets, and generated results are intended to support learning, research, experimentation, and engineering analysis.',
      'Results should be interpreted together with the limitations of the available dataset, model evaluation, experimental variability, and applicable engineering standards.',
    ],
  },
  {
    title: '10. Intellectual Property',
    content: [
      'The PrintOpt AI interface, software implementation, documentation, and project materials are part of an academic engineering project.',
      'Third-party research data and publications remain subject to their respective authors, publishers, institutions, licenses, and applicable intellectual-property rights.',
      'Nothing on this platform should be interpreted as transferring ownership of third-party research material.',
    ],
  },
  {
    title: '11. Changes to This Policy',
    content: [
      'This Privacy Policy may be updated when the PrintOpt AI platform, its functionality, data handling practices, or deployment environment changes.',
      'The latest version published on this page should be considered the current version of the policy.',
    ],
  },
  {
    title: '12. Contact and Project Attribution',
    content: [
      'PrintOpt AI is an academic project developed by Hardik Sonu and Samreen Utta Ur Rehman under the supervision of Dr. Ing. Waseem Amin at the University of the Punjab, Lahore.',
      'For project-related academic or technical inquiries, the project team or supervising academic unit should be contacted through the appropriate institutional channel.',
    ],
  },
];

export default function PrivacyPolicyPage() {
  return (
    <div className="max-w-[900px] mx-auto">

      {/* Header */}
      <div className="border-b border-[#1f2229] pb-8 mb-8">
        <Link
          href="/"
          className="inline-flex items-center gap-2 text-[11px] text-[#626873] hover:text-[#76B900] transition-colors mb-8"
        >
          <ArrowLeft size={12} />
          Back to PrintOpt AI
        </Link>

        <div className="flex items-start gap-4">
          <div className="flex items-center justify-center w-10 h-10 rounded bg-[#76B900]/10 border border-[#76B900]/20 shrink-0">
            <ShieldCheck
              size={19}
              className="text-[#76B900]"
            />
          </div>

          <div>
            <p className="text-[10px] uppercase tracking-[0.16em] text-[#76B900] mb-2">
              PrintOpt AI
            </p>

            <h1 className="text-3xl font-semibold tracking-tight text-[#f0f2f5]">
              Privacy Policy
            </h1>

            <p className="mt-3 text-[12px] leading-relaxed text-[#626873] max-w-2xl">
              Information about data handling, research data,
              machine-learning processing, and the academic
              purpose of the PrintOpt AI platform.
            </p>
          </div>
        </div>

        <div className="mt-6 flex flex-wrap items-center gap-x-5 gap-y-2 text-[10px] text-[#4b505b]">
          <span>Effective date: September 2026</span>
          <span>·</span>
          <span>Academic research project</span>
          <span>·</span>
          <span>University of the Punjab, Lahore</span>
        </div>
      </div>

      {/* Notice */}
      <div className="mb-8 rounded border border-[#76B900]/15 bg-[#76B900]/[0.035] px-5 py-4">
        <div className="text-[11px] font-medium text-[#aeb4bf] mb-1">
          Important
        </div>

        <p className="text-[11px] leading-relaxed text-[#626873]">
          PrintOpt AI is an academic research platform. Its
          machine-learning predictions and optimization results
          should be experimentally validated before being used
          for manufacturing, qualification, or safety-critical
          decisions.
        </p>
      </div>

      {/* Policy */}
      <div className="space-y-9">
        {sections.map((section) => (
          <section key={section.title}>
            <h2 className="text-[14px] font-semibold text-[#d5d9e0] mb-3">
              {section.title}
            </h2>

            <div className="space-y-2.5">
              {section.content.map((paragraph, index) => (
                <p
                  key={index}
                  className="text-[12px] leading-7 text-[#737985]"
                >
                  {paragraph}
                </p>
              ))}
            </div>
          </section>
        ))}
      </div>

      {/* Attribution */}
      <div className="mt-12 pt-6 border-t border-[#1f2229]">
        <div className="text-[10px] uppercase tracking-[0.14em] text-[#4b505b] mb-3">
          Project Attribution
        </div>

        <div className="text-[12px] text-[#8b909a]">
          PrintOpt AI
        </div>

        <div className="mt-1 text-[11px] text-[#5a5f6b]">
          Developed by Hardik Sonu and Samreen Utta Ur Rehman
        </div>

        <div className="text-[11px] text-[#5a5f6b]">
          Supervised by Dr. Ing. Waseem Amin
        </div>

        <div className="text-[11px] text-[#5a5f6b]">
          University of the Punjab, Lahore
        </div>

        <div className="mt-5 text-[10px] text-[#3f444d]">
          © 2026 PrintOpt AI. All rights reserved.
        </div>
      </div>
    </div>
  );
}