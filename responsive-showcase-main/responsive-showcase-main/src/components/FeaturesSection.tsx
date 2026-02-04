import { Brain, Zap, Shield, Cpu, Eye, BarChart3 } from 'lucide-react';

const features = [
  {
    icon: Brain,
    title: 'Deep Learning CNN',
    description: 'Advanced Convolutional Neural Networks trained on millions of facial expressions for accurate emotion classification.',
  },
  {
    icon: Zap,
    title: 'Real-Time Processing',
    description: 'Ultra-low latency inference enabling smooth real-time emotion detection from live video streams.',
  },
  {
    icon: Eye,
    title: 'Computer Vision',
    description: 'State-of-the-art face detection and preprocessing using OpenCV for robust feature extraction.',
  },
  {
    icon: Shield,
    title: 'Privacy First',
    description: 'All processing happens locally. No facial data is stored or transmitted to external servers.',
  },
  {
    icon: Cpu,
    title: 'Optimized Model',
    description: 'Lightweight architecture optimized for edge deployment without sacrificing accuracy.',
  },
  {
    icon: BarChart3,
    title: 'Detailed Analytics',
    description: 'Comprehensive emotion metrics and confidence scores for each detection result.',
  },
];

const FeaturesSection = () => {
  return (
    <section id="features" className="py-24 relative overflow-hidden">
      {/* Background Elements */}
      <div className="absolute top-0 left-0 w-full h-32 bg-gradient-to-b from-background to-transparent" />
      <div className="absolute bottom-0 left-0 w-full h-32 bg-gradient-to-t from-background to-transparent" />
      <div className="absolute inset-0 bg-secondary/30" />

      <div className="container mx-auto px-4 relative z-10">
        {/* Section Header */}
        <div className="text-center mb-16">
          <h2 className="font-display text-3xl md:text-5xl font-bold mb-4">
            Powerful <span className="text-gradient">Features</span>
          </h2>
          <p className="text-muted-foreground max-w-2xl mx-auto text-lg">
            Built with cutting-edge technology stack combining computer vision 
            and deep learning for accurate emotion recognition.
          </p>
        </div>

        {/* Features Grid */}
        <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-6">
          {features.map((feature, index) => (
            <div
              key={feature.title}
              className="group glass-card rounded-2xl p-8 border border-border/50 hover:border-primary/50 transition-all duration-500 hover:-translate-y-2 animate-fade-in-up"
              style={{ animationDelay: `${index * 100}ms` }}
            >
              {/* Icon Container */}
              <div className="relative mb-6">
                <div className="w-16 h-16 rounded-2xl bg-primary/10 flex items-center justify-center group-hover:bg-primary/20 transition-colors duration-300">
                  <feature.icon className="w-8 h-8 text-primary" />
                </div>
                <div className="absolute inset-0 bg-primary/20 blur-xl rounded-full opacity-0 group-hover:opacity-100 transition-opacity duration-500" />
              </div>

              {/* Content */}
              <h3 className="font-display text-xl font-semibold mb-3 text-foreground group-hover:text-primary transition-colors duration-300">
                {feature.title}
              </h3>
              <p className="text-muted-foreground leading-relaxed">
                {feature.description}
              </p>
            </div>
          ))}
        </div>
      </div>
    </section>
  );
};

export default FeaturesSection;
