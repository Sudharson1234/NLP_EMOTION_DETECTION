import Header from '@/components/Header';
import HeroSection from '@/components/HeroSection';
import DemoSection from '@/components/DemoSection';
import FeaturesSection from '@/components/FeaturesSection';
import HowItWorksSection from '@/components/HowItWorksSection';
import ApplicationsSection from '@/components/ApplicationsSection';
import Footer from '@/components/Footer';
import NeuralBackground from '@/components/NeuralBackground';

const Index = () => {
  return (
    <div className="min-h-screen bg-background overflow-x-hidden">
      {/* Neural Network Animated Background */}
      <NeuralBackground />
      
      {/* Header */}
      <Header />
      
      {/* Main Content */}
      <main className="relative z-10">
        <HeroSection />
        <DemoSection />
        <FeaturesSection />
        <HowItWorksSection />
        <ApplicationsSection />
      </main>
      
      {/* Footer */}
      <Footer />
    </div>
  );
};

export default Index;
