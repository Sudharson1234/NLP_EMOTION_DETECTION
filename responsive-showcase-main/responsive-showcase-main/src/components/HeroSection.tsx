import { ArrowRight, Play, Sparkles } from 'lucide-react';
import { Button } from './ui/button';

const HeroSection = () => {
  return (
    <section id="home" className="relative min-h-screen flex items-center justify-center pt-20 overflow-hidden">
      {/* Gradient Orbs */}
      <div className="absolute top-1/4 left-1/4 w-96 h-96 bg-primary/20 rounded-full blur-3xl animate-pulse-glow" />
      <div className="absolute bottom-1/4 right-1/4 w-96 h-96 bg-accent/20 rounded-full blur-3xl animate-pulse-glow animation-delay-500" />
      
      <div className="container mx-auto px-4 relative z-10">
        <div className="max-w-5xl mx-auto text-center">
          {/* Badge */}
          <div className="inline-flex items-center gap-2 px-4 py-2 rounded-full glass-card border border-primary/30 mb-8 animate-fade-in-up">
            <Sparkles className="w-4 h-4 text-primary" />
            <span className="text-sm text-muted-foreground">AI-Powered Emotion Recognition</span>
          </div>

          {/* Main Heading */}
          <h1 className="font-display text-4xl md:text-6xl lg:text-7xl font-bold mb-6 leading-tight animate-fade-in-up animation-delay-100">
            Real-Time{' '}
            <span className="text-gradient">Emotion Detection</span>
            <br />
            from Facial Expressions
          </h1>

          {/* Subtitle */}
          <p className="text-lg md:text-xl text-muted-foreground max-w-3xl mx-auto mb-10 leading-relaxed animate-fade-in-up animation-delay-200">
            Harness the power of Computer Vision and Deep Learning to understand 
            human emotions through visual cues. Detect happiness, sadness, anger, 
            fear, surprise, and neutral expressions in real-time.
          </p>

          {/* CTA Buttons */}
          <div className="flex flex-col sm:flex-row items-center justify-center gap-4 animate-fade-in-up animation-delay-300">
            <Button 
              size="lg" 
              className="bg-primary text-primary-foreground hover:bg-primary/90 glow-box font-semibold text-lg px-8 py-6 group"
              asChild
            >
              <a href="#demo" className="flex items-center gap-2">
                Start Detection
                <ArrowRight className="w-5 h-5 group-hover:translate-x-1 transition-transform" />
              </a>
            </Button>
            <Button 
              size="lg" 
              variant="outline" 
              className="border-border bg-transparent hover:bg-secondary text-foreground font-semibold text-lg px-8 py-6 group"
              asChild
            >
              <a href="#features" className="flex items-center gap-2">
                <Play className="w-5 h-5 mr-2 group-hover:scale-110 transition-transform" />
                Learn More
              </a>
            </Button>
          </div>

          {/* Stats */}
          <div className="grid grid-cols-2 md:grid-cols-4 gap-6 mt-16 animate-fade-in-up animation-delay-400">
            {[
              { value: '6', label: 'Emotions' },
              { value: '95%+', label: 'Accuracy' },
              { value: '<50ms', label: 'Latency' },
              { value: 'Real-Time', label: 'Processing' },
            ].map((stat) => (
              <div key={stat.label} className="glass-card rounded-xl p-6 border border-border/50">
                <div className="font-display text-2xl md:text-3xl font-bold text-gradient mb-2">
                  {stat.value}
                </div>
                <div className="text-sm text-muted-foreground">{stat.label}</div>
              </div>
            ))}
          </div>
        </div>
      </div>

      {/* Scroll Indicator */}
      <div className="absolute bottom-8 left-1/2 -translate-x-1/2 animate-float">
        <div className="w-6 h-10 rounded-full border-2 border-primary/50 flex justify-center p-2">
          <div className="w-1 h-2 bg-primary rounded-full animate-pulse" />
        </div>
      </div>
    </section>
  );
};

export default HeroSection;
