import { Brain, Github, Linkedin, Mail, ExternalLink } from 'lucide-react';

const Footer = () => {
  return (
    <footer className="py-16 border-t border-border/50 relative">
      <div className="container mx-auto px-4">
        <div className="grid md:grid-cols-2 lg:grid-cols-4 gap-12">
          {/* Brand */}
          <div className="lg:col-span-2">
            <div className="flex items-center gap-2 mb-4">
              <Brain className="w-8 h-8 text-primary" />
              <span className="font-display text-xl font-bold text-gradient">EmotiSense</span>
            </div>
            <p className="text-muted-foreground max-w-md mb-6">
              Real-time emotion detection system using computer vision and deep learning. 
              A final year project showcasing the power of AI in understanding human emotions.
            </p>
            <div className="flex gap-4">
              {[
                { icon: Github, href: '#' },
                { icon: Linkedin, href: '#' },
                { icon: Mail, href: '#' },
              ].map((social, index) => (
                <a
                  key={index}
                  href={social.href}
                  className="w-10 h-10 rounded-lg bg-secondary flex items-center justify-center hover:bg-primary/20 hover:text-primary transition-all duration-300"
                >
                  <social.icon className="w-5 h-5" />
                </a>
              ))}
            </div>
          </div>

          {/* Quick Links */}
          <div>
            <h4 className="font-display font-semibold mb-4 text-foreground">Quick Links</h4>
            <ul className="space-y-3">
              {['Home', 'Demo', 'Features', 'How It Works', 'Applications'].map((link) => (
                <li key={link}>
                  <a
                    href={`#${link.toLowerCase().replace(' ', '-')}`}
                    className="text-muted-foreground hover:text-primary transition-colors duration-300 flex items-center gap-2"
                  >
                    <ExternalLink className="w-3 h-3" />
                    {link}
                  </a>
                </li>
              ))}
            </ul>
          </div>

          {/* Project Info */}
          <div>
            <h4 className="font-display font-semibold mb-4 text-foreground">Project Info</h4>
            <ul className="space-y-3 text-muted-foreground">
              <li>
                <span className="text-foreground">Type:</span> Final Year Project
              </li>
              <li>
                <span className="text-foreground">Domain:</span> AI/ML & Computer Vision
              </li>
              <li>
                <span className="text-foreground">Technologies:</span> TensorFlow, OpenCV, React
              </li>
              <li>
                <span className="text-foreground">Year:</span> 2024-2025
              </li>
            </ul>
          </div>
        </div>

        {/* Bottom Bar */}
        <div className="mt-12 pt-8 border-t border-border/50 flex flex-col md:flex-row justify-between items-center gap-4">
          <p className="text-sm text-muted-foreground">
            © 2024 EmotiSense. All rights reserved.
          </p>
          <p className="text-sm text-muted-foreground">
            Final Year Project - Computer Science & Engineering
          </p>
        </div>
      </div>
    </footer>
  );
};

export default Footer;
