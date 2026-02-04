import { Camera, Scan, Brain, BarChart } from 'lucide-react';

const steps = [
  {
    icon: Camera,
    number: '01',
    title: 'Image Capture',
    description: 'Real-time video stream or static image is captured through the camera interface.',
  },
  {
    icon: Scan,
    number: '02',
    title: 'Face Detection',
    description: 'OpenCV identifies and isolates facial regions using Haar Cascade classifiers.',
  },
  {
    icon: Brain,
    number: '03',
    title: 'CNN Analysis',
    description: 'Preprocessed facial features are analyzed by our trained deep learning model.',
  },
  {
    icon: BarChart,
    number: '04',
    title: 'Emotion Output',
    description: 'Classification results with confidence scores are displayed in real-time.',
  },
];

const HowItWorksSection = () => {
  return (
    <section id="how-it-works" className="py-24 relative">
      <div className="container mx-auto px-4">
        {/* Section Header */}
        <div className="text-center mb-16">
          <h2 className="font-display text-3xl md:text-5xl font-bold mb-4">
            How It <span className="text-gradient">Works</span>
          </h2>
          <p className="text-muted-foreground max-w-2xl mx-auto text-lg">
            A streamlined pipeline from image capture to emotion classification 
            using state-of-the-art deep learning techniques.
          </p>
        </div>

        {/* Steps */}
        <div className="relative">
          {/* Connection Line */}
          <div className="hidden lg:block absolute top-1/2 left-0 right-0 h-0.5 bg-gradient-to-r from-transparent via-primary/30 to-transparent -translate-y-1/2" />

          <div className="grid md:grid-cols-2 lg:grid-cols-4 gap-8">
            {steps.map((step, index) => (
              <div
                key={step.number}
                className="relative group animate-fade-in-up"
                style={{ animationDelay: `${index * 150}ms` }}
              >
                {/* Card */}
                <div className="glass-card rounded-2xl p-8 border border-border/50 hover:border-primary/50 transition-all duration-500 h-full">
                  {/* Number Badge */}
                  <div className="absolute -top-4 left-8 px-4 py-1 bg-primary text-primary-foreground font-display font-bold text-sm rounded-full glow-box">
                    {step.number}
                  </div>

                  {/* Icon */}
                  <div className="w-20 h-20 rounded-2xl bg-gradient-to-br from-primary/20 to-accent/20 flex items-center justify-center mb-6 group-hover:scale-110 transition-transform duration-500">
                    <step.icon className="w-10 h-10 text-primary" />
                  </div>

                  {/* Content */}
                  <h3 className="font-display text-xl font-semibold mb-3 text-foreground">
                    {step.title}
                  </h3>
                  <p className="text-muted-foreground leading-relaxed">
                    {step.description}
                  </p>
                </div>

                {/* Arrow Connector */}
                {index < steps.length - 1 && (
                  <div className="hidden lg:block absolute top-1/2 -right-4 w-8 h-8 -translate-y-1/2 z-10">
                    <div className="w-full h-full border-t-2 border-r-2 border-primary/50 rotate-45 animate-pulse" />
                  </div>
                )}
              </div>
            ))}
          </div>
        </div>

        {/* Technical Stack */}
        <div className="mt-20 glass-card rounded-3xl p-8 md:p-12 border border-border/50">
          <h3 className="font-display text-2xl font-bold mb-8 text-center">
            Technology Stack
          </h3>
          <div className="grid grid-cols-2 md:grid-cols-4 gap-6">
            {[
              { name: 'TensorFlow', type: 'Deep Learning' },
              { name: 'OpenCV', type: 'Computer Vision' },
              { name: 'Python', type: 'Backend' },
              { name: 'React', type: 'Frontend' },
            ].map((tech, index) => (
              <div
                key={tech.name}
                className="text-center p-6 rounded-xl bg-secondary/50 hover:bg-secondary transition-colors duration-300 animate-fade-in-up"
                style={{ animationDelay: `${index * 100}ms` }}
              >
                <div className="font-display font-bold text-lg text-primary mb-1">
                  {tech.name}
                </div>
                <div className="text-sm text-muted-foreground">{tech.type}</div>
              </div>
            ))}
          </div>
        </div>
      </div>
    </section>
  );
};

export default HowItWorksSection;
