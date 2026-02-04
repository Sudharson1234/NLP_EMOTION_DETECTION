import { LucideIcon } from 'lucide-react';

interface EmotionCardProps {
  name: string;
  icon: LucideIcon;
  percentage: number;
  color: string;
  description: string;
  isActive?: boolean;
  delay?: number;
}

const EmotionCard = ({ 
  name, 
  icon: Icon, 
  percentage, 
  color, 
  description,
  isActive = false,
  delay = 0 
}: EmotionCardProps) => {
  return (
    <div 
      className={`glass-card rounded-2xl p-6 border transition-all duration-500 hover:scale-105 hover:border-primary/50 animate-fade-in-up ${
        isActive ? 'border-primary glow-box' : 'border-border/50'
      }`}
      style={{ 
        animationDelay: `${delay}ms`,
        '--bar-width': `${percentage}%`
      } as React.CSSProperties}
    >
      {/* Icon */}
      <div 
        className={`w-14 h-14 rounded-xl flex items-center justify-center mb-4 ${color}`}
      >
        <Icon className="w-7 h-7" />
      </div>

      {/* Name */}
      <h3 className="font-display text-xl font-semibold mb-2 text-foreground">
        {name}
      </h3>

      {/* Description */}
      <p className="text-sm text-muted-foreground mb-4 line-clamp-2">
        {description}
      </p>

      {/* Progress Bar */}
      <div className="relative">
        <div className="h-2 bg-secondary rounded-full overflow-hidden">
          <div 
            className={`h-full rounded-full animate-emotion-bar ${color.replace('bg-', 'bg-').replace('/20', '')}`}
            style={{ width: `${percentage}%` }}
          />
        </div>
        <div className="flex justify-between mt-2">
          <span className="text-xs text-muted-foreground">Confidence</span>
          <span className="text-xs font-semibold text-primary">{percentage}%</span>
        </div>
      </div>
    </div>
  );
};

export default EmotionCard;
