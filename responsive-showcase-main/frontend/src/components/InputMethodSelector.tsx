import { Camera, Upload, Video } from 'lucide-react';
import { Button } from './ui/button';

type InputMethod = 'camera' | 'image' | 'video';

interface InputMethodSelectorProps {
  selectedMethod: InputMethod;
  onMethodChange: (method: InputMethod) => void;
  disabled?: boolean;
}

const InputMethodSelector = ({ selectedMethod, onMethodChange, disabled }: InputMethodSelectorProps) => {
  const methods = [
    { id: 'camera' as InputMethod, label: 'Camera', icon: Camera, description: 'Real-time detection' },
    { id: 'image' as InputMethod, label: 'Image', icon: Upload, description: 'Upload photo' },
    { id: 'video' as InputMethod, label: 'Video', icon: Video, description: 'Upload video' },
  ];

  return (
    <div className="flex gap-2 p-1 bg-secondary/50 rounded-xl">
      {methods.map((method) => (
        <Button
          key={method.id}
          variant={selectedMethod === method.id ? 'default' : 'ghost'}
          className={`flex-1 flex-col h-auto py-3 gap-1 rounded-lg transition-all ${
            selectedMethod === method.id 
              ? 'bg-primary text-primary-foreground shadow-lg' 
              : 'hover:bg-secondary'
          }`}
          onClick={() => onMethodChange(method.id)}
          disabled={disabled}
        >
          <method.icon className="w-5 h-5" />
          <span className="text-xs font-medium">{method.label}</span>
        </Button>
      ))}
    </div>
  );
};

export default InputMethodSelector;
