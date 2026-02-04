import { Monitor, ShieldCheck, Heart, ShoppingCart, GraduationCap, Users } from 'lucide-react';

const applications = [
  {
    icon: Monitor,
    title: 'Human-Computer Interaction',
    description: 'Enable adaptive interfaces that respond to user emotional states for enhanced user experience.',
    gradient: 'from-blue-500/20 to-cyan-500/20',
  },
  {
    icon: ShieldCheck,
    title: 'Surveillance & Security',
    description: 'Detect suspicious behavior and emotional distress in public spaces for improved safety.',
    gradient: 'from-red-500/20 to-orange-500/20',
  },
  {
    icon: Heart,
    title: 'Mental Health Monitoring',
    description: 'Track emotional patterns to assist in therapy and early detection of mental health issues.',
    gradient: 'from-pink-500/20 to-rose-500/20',
  },
  {
    icon: ShoppingCart,
    title: 'Customer Behavior Analysis',
    description: 'Understand customer reactions to products and services for better business insights.',
    gradient: 'from-green-500/20 to-emerald-500/20',
  },
  {
    icon: GraduationCap,
    title: 'Education & E-Learning',
    description: 'Adapt learning content based on student engagement and emotional response.',
    gradient: 'from-purple-500/20 to-violet-500/20',
  },
  {
    icon: Users,
    title: 'Social Robotics',
    description: 'Enable robots to understand and respond appropriately to human emotions.',
    gradient: 'from-yellow-500/20 to-amber-500/20',
  },
];

const ApplicationsSection = () => {
  return (
    <section id="applications" className="py-24 relative overflow-hidden">
      {/* Background Pattern */}
      <div className="absolute inset-0 opacity-5">
        <div className="absolute inset-0" style={{
          backgroundImage: 'radial-gradient(circle at 2px 2px, currentColor 1px, transparent 0)',
          backgroundSize: '40px 40px',
        }} />
      </div>

      <div className="container mx-auto px-4 relative z-10">
        {/* Section Header */}
        <div className="text-center mb-16">
          <h2 className="font-display text-3xl md:text-5xl font-bold mb-4">
            Real-World <span className="text-gradient">Applications</span>
          </h2>
          <p className="text-muted-foreground max-w-2xl mx-auto text-lg">
            Emotion detection technology is transforming industries and 
            creating new possibilities for human-machine interaction.
          </p>
        </div>

        {/* Applications Grid */}
        <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-6">
          {applications.map((app, index) => (
            <div
              key={app.title}
              className="group relative glass-card rounded-2xl p-8 border border-border/50 hover:border-primary/50 transition-all duration-500 overflow-hidden animate-fade-in-up"
              style={{ animationDelay: `${index * 100}ms` }}
            >
              {/* Gradient Background */}
              <div className={`absolute inset-0 bg-gradient-to-br ${app.gradient} opacity-0 group-hover:opacity-100 transition-opacity duration-500`} />
              
              {/* Content */}
              <div className="relative z-10">
                {/* Icon */}
                <div className="w-14 h-14 rounded-xl bg-primary/10 flex items-center justify-center mb-6 group-hover:scale-110 group-hover:bg-primary/20 transition-all duration-300">
                  <app.icon className="w-7 h-7 text-primary" />
                </div>

                <h3 className="font-display text-xl font-semibold mb-3 text-foreground group-hover:text-primary transition-colors duration-300">
                  {app.title}
                </h3>
                <p className="text-muted-foreground leading-relaxed">
                  {app.description}
                </p>
              </div>

              {/* Hover Glow */}
              <div className="absolute bottom-0 left-0 right-0 h-1 bg-gradient-to-r from-transparent via-primary to-transparent opacity-0 group-hover:opacity-100 transition-opacity duration-500" />
            </div>
          ))}
        </div>

        {/* Future Enhancements */}
        <div className="mt-20 text-center">
          <h3 className="font-display text-2xl font-bold mb-8">
            Future Enhancements
          </h3>
          <div className="flex flex-wrap justify-center gap-4">
            {[
              'Multi-Face Tracking',
              'Emotion Intensity Estimation',
              'Cross-Cultural Adaptation',
              'Voice Emotion Fusion',
              'Edge Device Deployment',
            ].map((enhancement, index) => (
              <div
                key={enhancement}
                className="px-6 py-3 glass-card rounded-full border border-border/50 hover:border-primary/50 hover:glow-box transition-all duration-300 animate-fade-in-up"
                style={{ animationDelay: `${index * 100}ms` }}
              >
                <span className="text-sm font-medium text-muted-foreground hover:text-primary transition-colors">
                  {enhancement}
                </span>
              </div>
            ))}
          </div>
        </div>
      </div>
    </section>
  );
};

export default ApplicationsSection;
