import { ShieldCheck, Microscope, BrainCircuit } from "lucide-react";

export default function About() {
  return (
    <>
      <div className="page-title">
        <div>
          <span className="eyebrow">PROJECT INFORMATION</span>
          <h1>About OncoSight AI</h1>
          <p>A modular medical-imaging machine-learning research prototype.</p>
        </div>
      </div>

      <section className="panel">
        <div className="about-grid">
          <div>
            <Microscope />
            <h3>Research workflow</h3>
            <p>
              Organize sample imaging studies, inspect previews and review model
              outputs in one interface.
            </p>
          </div>

          <div>
            <BrainCircuit />
            <h3>Machine learning</h3>
            <p>
              Designed to connect to experimental classifiers and image-analysis
              models through a backend API.
            </p>
          </div>

          <div>
            <ShieldCheck />
            <h3>Responsible use</h3>
            <p>
              Outputs require validation, bias evaluation and qualified human
              review before any real-world application.
            </p>
          </div>
        </div>
      </section>

      <div className="notice">
        <b>Important:</b> OncoSight AI is an educational/research prototype. It
        is not a medical device and must not be used for diagnosis, screening,
        treatment or clinical decisions.
      </div>
    </>
  );
}
