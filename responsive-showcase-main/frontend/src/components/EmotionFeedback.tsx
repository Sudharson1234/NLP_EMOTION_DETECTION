import { useState, useEffect } from 'react';
import { Heart, MessageCircle, Sparkles, Send, X } from 'lucide-react';
import { Button } from './ui/button';

interface EmotionFeedbackProps {
  emotion: string | null;
  confidence: number;
  onClose: () => void;
}

const emotionResponses: Record<string, {
  greeting: string;
  question: string;
  responses: string[];
  encouragement: string[];
  color: string;
}> = {
  happy: {
    greeting: "You look happy! 😊",
    question: "What's making you smile today?",
    responses: [
      "That's wonderful! Keep spreading those positive vibes!",
      "Your happiness is contagious! Keep it going!",
      "It's great to see you in such high spirits!",
    ],
    encouragement: [
      "Remember this feeling when times get tough!",
      "Share your joy with someone else today!",
      "Happiness looks great on you!",
    ],
    color: 'text-emotion-happy',
  },
  sad: {
    greeting: "I notice you seem a bit down 💙",
    question: "Would you like to share what's troubling you?",
    responses: [
      "I'm here for you. It's okay to feel this way.",
      "Thank you for sharing. Remember, this feeling is temporary.",
      "I understand. Sometimes we all need a moment to feel our emotions.",
    ],
    encouragement: [
      "Remember, after every storm comes a rainbow. You've got this! 🌈",
      "You are stronger than you think. Take it one step at a time. 💪",
      "It's okay to not be okay. Be gentle with yourself today. 🤗",
      "This too shall pass. You have overcome challenges before and you will again.",
      "Take a deep breath. You are loved and you matter. ❤️",
    ],
    color: 'text-emotion-sad',
  },
  angry: {
    greeting: "I can see you might be feeling frustrated 🔥",
    question: "What's on your mind? Sometimes talking helps.",
    responses: [
      "I hear you. Those feelings are valid.",
      "It's okay to feel angry. Let's work through this together.",
      "Thank you for expressing yourself. That takes courage.",
    ],
    encouragement: [
      "Try taking 3 deep breaths. Inhale... exhale... 🧘",
      "Consider stepping away for a moment - a short walk can help clear your mind.",
      "Remember, you have the power to choose your response. You've got this!",
      "Channel this energy into something positive - maybe exercise or creative work?",
    ],
    color: 'text-emotion-angry',
  },
  fearful: {
    greeting: "You seem a bit anxious or worried 🫂",
    question: "What's causing you concern?",
    responses: [
      "Thank you for trusting me. Fear is a natural response.",
      "I understand. Let's talk through what's worrying you.",
      "You're brave for acknowledging your feelings.",
    ],
    encouragement: [
      "You are safe right now. Take a moment to ground yourself. 🌿",
      "Fear is just a feeling - it doesn't define you or your capabilities.",
      "Try the 5-4-3-2-1 technique: notice 5 things you see, 4 you hear, 3 you touch...",
      "Whatever happens, you have the strength to handle it. Believe in yourself! 💫",
    ],
    color: 'text-emotion-fear',
  },
  surprised: {
    greeting: "Something caught you off guard! ⚡",
    question: "Was it a good surprise or an unexpected one?",
    responses: [
      "Life is full of surprises! How are you processing this?",
      "Unexpected moments can be exciting or overwhelming. How do you feel?",
      "Surprises keep life interesting! Tell me more about it.",
    ],
    encouragement: [
      "Embrace the unexpected - it often leads to new opportunities!",
      "Stay curious and open to what life brings your way! ✨",
      "Sometimes the best things in life are unplanned!",
    ],
    color: 'text-emotion-surprise',
  },
  neutral: {
    greeting: "You seem calm and balanced ☯️",
    question: "How's your day going so far?",
    responses: [
      "A balanced state of mind is valuable. How can I help today?",
      "Sometimes neutral is the perfect place to be.",
      "Serenity is a gift. Enjoy this moment of peace.",
    ],
    encouragement: [
      "This calm state is perfect for reflection or planning ahead.",
      "Balance is key to well-being. You're doing great! 🎯",
      "Use this peaceful moment to check in with yourself.",
    ],
    color: 'text-emotion-neutral',
  },
  disgusted: {
    greeting: "Something seems to be bothering you 😕",
    question: "Would you like to talk about what's causing this feeling?",
    responses: [
      "I understand. Sometimes things can be upsetting.",
      "Those feelings are valid. Let's work through this.",
      "Thank you for sharing. What would help you feel better?",
    ],
    encouragement: [
      "Focus on what brings you joy and peace instead.",
      "Try to shift your attention to something positive.",
      "Remember, you have control over what you give your energy to.",
    ],
    color: 'text-emotion-neutral',
  },
};

const EmotionFeedback = ({ emotion, confidence, onClose }: EmotionFeedbackProps) => {
  const [stage, setStage] = useState<'greeting' | 'question' | 'response' | 'encouragement'>('greeting');
  const [userMessage, setUserMessage] = useState('');
  const [messages, setMessages] = useState<Array<{ type: 'bot' | 'user'; text: string }>>([]);
  const [isTyping, setIsTyping] = useState(false);

  const emotionData = emotion ? emotionResponses[emotion.toLowerCase()] || emotionResponses.neutral : emotionResponses.neutral;

  useEffect(() => {
    if (emotion) {
      setMessages([{ type: 'bot', text: emotionData.greeting }]);
      setStage('greeting');
      
      // Auto-advance to question after greeting
      const timer = setTimeout(() => {
        setIsTyping(true);
        setTimeout(() => {
          setMessages(prev => [...prev, { type: 'bot', text: emotionData.question }]);
          setIsTyping(false);
          setStage('question');
        }, 1000);
      }, 1500);
      
      return () => clearTimeout(timer);
    }
  }, [emotion, emotionData.greeting, emotionData.question]);

  const handleSendMessage = () => {
    if (!userMessage.trim()) return;

    const newUserMessage = { type: 'user' as const, text: userMessage };
    setMessages(prev => [...prev, newUserMessage]);
    setUserMessage('');
    setIsTyping(true);

    // Simulate bot response
    setTimeout(() => {
      const response = emotionData.responses[Math.floor(Math.random() * emotionData.responses.length)];
      setMessages(prev => [...prev, { type: 'bot', text: response }]);
      setIsTyping(false);
      setStage('response');

      // Add encouragement after response
      setTimeout(() => {
        setIsTyping(true);
        setTimeout(() => {
          const encouragement = emotionData.encouragement[Math.floor(Math.random() * emotionData.encouragement.length)];
          setMessages(prev => [...prev, { type: 'bot', text: encouragement }]);
          setIsTyping(false);
          setStage('encouragement');
        }, 1200);
      }, 1500);
    }, 1500);
  };

  const handleKeyPress = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSendMessage();
    }
  };

  return (
    <div className="glass-card rounded-2xl p-6 border border-border/50 animate-fade-in-up">
      <div className="flex items-center justify-between mb-4">
        <div className="flex items-center gap-3">
          <div className="p-2 rounded-xl bg-primary/20">
            <MessageCircle className="w-5 h-5 text-primary" />
          </div>
          <div>
            <h3 className="font-display font-bold text-lg">Emotional Support</h3>
            <p className="text-xs text-muted-foreground">
              Detected: <span className={emotionData.color}>{emotion}</span> ({confidence}% confidence)
            </p>
          </div>
        </div>
        <Button variant="ghost" size="icon" onClick={onClose} className="h-8 w-8">
          <X className="w-4 h-4" />
        </Button>
      </div>

      {/* Chat Messages */}
      <div className="h-64 overflow-y-auto mb-4 space-y-3 pr-2 scrollbar-thin scrollbar-thumb-primary/20">
        {messages.map((msg, index) => (
          <div
            key={index}
            className={`flex ${msg.type === 'user' ? 'justify-end' : 'justify-start'}`}
          >
            <div
              className={`max-w-[85%] rounded-2xl px-4 py-3 ${
                msg.type === 'user'
                  ? 'bg-primary text-primary-foreground rounded-br-sm'
                  : 'bg-secondary/80 text-foreground rounded-bl-sm'
              }`}
            >
              {msg.type === 'bot' && (
                <Sparkles className="w-4 h-4 inline mr-2 text-primary" />
              )}
              <span className="text-sm">{msg.text}</span>
            </div>
          </div>
        ))}
        {isTyping && (
          <div className="flex justify-start">
            <div className="bg-secondary/80 rounded-2xl rounded-bl-sm px-4 py-3">
              <div className="flex gap-1">
                <span className="w-2 h-2 bg-primary/60 rounded-full animate-bounce" style={{ animationDelay: '0ms' }} />
                <span className="w-2 h-2 bg-primary/60 rounded-full animate-bounce" style={{ animationDelay: '150ms' }} />
                <span className="w-2 h-2 bg-primary/60 rounded-full animate-bounce" style={{ animationDelay: '300ms' }} />
              </div>
            </div>
          </div>
        )}
      </div>

      {/* Input Area */}
      <div className="flex gap-2">
        <input
          type="text"
          value={userMessage}
          onChange={(e) => setUserMessage(e.target.value)}
          onKeyPress={handleKeyPress}
          placeholder="Type your message..."
          className="flex-1 bg-secondary/50 border border-border/50 rounded-xl px-4 py-3 text-sm focus:outline-none focus:ring-2 focus:ring-primary/50 placeholder:text-muted-foreground"
        />
        <Button
          onClick={handleSendMessage}
          disabled={!userMessage.trim() || isTyping}
          className="bg-primary hover:bg-primary/90 rounded-xl px-4"
        >
          <Send className="w-4 h-4" />
        </Button>
      </div>

      {/* Quick Responses */}
      {stage === 'question' && (
        <div className="mt-3 flex flex-wrap gap-2">
          <Button
            variant="outline"
            size="sm"
            className="text-xs rounded-full border-border/50 hover:bg-primary/10"
            onClick={() => {
              setUserMessage("I'd like to talk about it");
              handleSendMessage();
            }}
          >
            <Heart className="w-3 h-3 mr-1" /> I'd like to talk
          </Button>
          <Button
            variant="outline"
            size="sm"
            className="text-xs rounded-full border-border/50 hover:bg-primary/10"
            onClick={() => {
              setUserMessage("Just checking in with myself");
            }}
          >
            Just checking in
          </Button>
        </div>
      )}
    </div>
  );
};

export default EmotionFeedback;
