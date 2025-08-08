import React, { useState, useEffect } from 'react';
import Marketplace from './Marketplace';
import NeuralNetworkVisualizer from './NeuralNetworkVisualizer';

const AIBusinessWebsite = () => {
  const [isVisible, setIsVisible] = useState({});
  const [showMarketPlace ,setShowMarketplace] = useState(false)
  const whatsappUrl = "https://wa.me/7507088360?text=Hi! I'm interested in your AI application development services.";

  const services = [
    { 
      icon: "🤖", 
      title: "Custom AI Applications", 
      desc: "Tailored AI solutions built specifically for your business needs" 
    },
    { 
      icon: "🧠", 
      title: "Machine Learning Models", 
      desc: "Advanced ML algorithms to predict, classify, and optimize your data" 
    },
    { 
      icon: "📊", 
      title: "Data Analytics Solutions", 
      desc: "Transform raw data into actionable business insights" 
    },
    { 
      icon: "🔌", 
      title: "Web Applications Development", 
      desc: "Robust APIs to integrate AI capabilities into your existing systems" 
    },
    { 
      icon: "💬", 
      title: "AI Chatbots", 
      desc: "Intelligent conversational agents for customer support and engagement" 
    },
    { 
      icon: "🎯", 
      title: "Predictive Analytics", 
      desc: "Forecast trends and make data-driven decisions with confidence" 
    },
    
  ];

  const platforms = [
    { name: "Vercel", icon: "▲", color: "from-black to-gray-800" },
    { name: "AWS", icon: "☁️", color: "from-orange-500 to-yellow-600" },
    { name: "Google Cloud", icon: "🌤️", color: "from-blue-500 to-green-500" },
    { name: "BlueHost", icon: "🏠", color: "from-blue-600 to-blue-800" },
    { name: "Heroku", icon: "🚀", color: "from-purple-600 to-pink-600" },
    { name: "Azure", icon: "☁️", color: "from-blue-400 to-blue-600" }
  ];

  const testimonials = [
    {
      text: "Exceptional AI solutions that transformed our business processes. Highly recommended!",
      author: "Tech Startup CEO"
    },
    {
      text: "Professional, efficient, and delivered exactly what we needed for our ML project.",
      author: "E-commerce Manager"
    }
  ];

  useEffect(() => {
    const observer = new IntersectionObserver(
      (entries) => {
        entries.forEach((entry) => {
          setIsVisible(prev => ({
            ...prev,
            [entry.target.id]: entry.isIntersecting
          }));
        });
      },
      { threshold: 0.1 }
    );

    document.querySelectorAll('[id]').forEach((el) => observer.observe(el));
    return () => observer.disconnect();
  }, []);

  return (
    <div className="min-h-screen bg-gray-50">
      {/* Navigation */}
      <nav className="bg-white shadow-lg sticky top-0 z-50">
        <div className="max-w-6xl mx-auto px-4">
          <div className="flex justify-between items-center py-4">
            <div className="flex items-center space-x-2">
              <span className="text-3xl">🤖</span>
              <span className="text-xl font-bold bg-gradient-to-r from-blue-600 to-purple-600 bg-clip-text text-transparent">
                AI Solutions
              </span>
            </div>
            <div className="hidden md:flex space-x-8">
              <a href="#home" className="text-gray-700 hover:text-blue-600 transition-colors">Home</a>
              <a href="#services" className="text-gray-700 hover:text-blue-600 transition-colors">Services</a>
              <a href="#about" className="text-gray-700 hover:text-blue-600 transition-colors">About</a>
              <a href="#contact" className="text-gray-700 hover:text-blue-600 transition-colors">Contact</a>
            </div>
            <a 
              href={whatsappUrl}
              target="_blank"
              rel="noopener noreferrer"
              className="bg-green-500 hover:bg-green-600 text-white px-4 py-2 rounded-full font-semibold transition-all duration-300 hover:scale-105"
            >
              💬 WhatsApp
            </a>
            
          </div>
        </div>
      </nav>
    
      {/* Hero Section */}
      <section id="home" className="bg-gradient-to-br from-blue-600 via-purple-700 to-indigo-800 text-white py-20">
        <div className="max-w-6xl mx-auto px-4 text-center">
          <div className="animate-bounce text-8xl mb-8">🤖</div>
          <h1 className="text-6xl md:text-7xl font-bold mb-6 animate-pulse">
            AI Application Development
          </h1>
          <p className="text-2xl md:text-3xl mb-8 opacity-90">
            End-to-End Solutions for Your Business
          </p>
          <div className="flex flex-col md:flex-row gap-4 justify-center">
            <a 
              href={whatsappUrl}
              target="_blank"
              rel="noopener noreferrer"
              className="bg-green-500 hover:bg-green-600 text-white px-8 py-4 rounded-full text-lg font-bold transition-all duration-300 hover:scale-105 shadow-lg"
            >
              💬 Get Started on WhatsApp
            </a>
            <a 
              href="#services"
              className="border-2 border-white text-white hover:bg-white hover:text-blue-600 px-8 py-4 rounded-full text-lg font-bold transition-all duration-300"
            >
              Explore Services
            </a>
            
            
          </div>
        </div>
      </section>
      {showMarketPlace && <Marketplace />}
      {/* About Section */}
      <section id="about" className="py-20 bg-white">
        <div className="max-w-6xl mx-auto px-4">
          <div className="text-center mb-16">
            <h2 className="text-5xl font-bold text-gray-800 mb-6">About Us</h2>
            <div className="w-24 h-1 bg-gradient-to-r from-blue-500 to-purple-500 mx-auto mb-8"></div>
          </div>
          
          <div className="bg-gradient-to-r from-blue-500 to-purple-600 text-white p-8 md:p-12 rounded-3xl shadow-2xl relative overflow-hidden">
            <div className="absolute inset-0 bg-gradient-to-r from-transparent via-white/10 to-transparent -translate-x-full animate-[shimmer_4s_infinite]"></div>
            <div className="relative z-10">
              <div className="grid md:grid-cols-3 gap-8 text-center">
                <div className="space-y-4">
                  <div className="text-4xl">🎓</div>
                  <h3 className="text-xl font-bold">Education</h3>
                  <p className="text-lg">We are a team of Data Scientists from IIT Madras</p>
                </div>
                <div className="space-y-4">
                  <div className="text-4xl">💻</div>
                  <h3 className="text-xl font-bold">Experience</h3>
                  <p className="text-lg">10+ Years Python Development</p>
                </div>
                <div className="space-y-4">
                  <div className="text-4xl">🚀</div>
                  <h3 className="text-xl font-bold">Specialization</h3>
                  <p className="text-lg">AI, Machine Learning & Programming Experts</p>
                </div>
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* Services Section */}
      <section id="services" className="py-20 bg-gray-50">
        <div className="max-w-6xl mx-auto px-4">
          <div className="text-center mb-16">
            <h2 className="text-5xl font-bold text-gray-800 mb-6">Our Services</h2>
            <div className="w-24 h-1 bg-gradient-to-r from-blue-500 to-purple-500 mx-auto mb-8"></div>
            <p className="text-xl text-gray-600 max-w-3xl mx-auto">
              Comprehensive AI solutions designed to transform your business operations and drive growth
            </p>
          </div>
          
          <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-8">
            {services.map((service, index) => (
              <div 
                key={index}
                className="bg-white p-8 rounded-3xl shadow-lg hover:shadow-2xl transition-all duration-500 hover:-translate-y-2 border-2 border-transparent hover:border-blue-200 group"
              >
                <div className="text-5xl mb-4 group-hover:animate-bounce">{service.icon}</div>
                <h3 className="text-xl font-bold text-gray-800 mb-4">{service.title}</h3>
                <p className="text-gray-600 leading-relaxed">{service.desc}</p>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* Platforms Section */}
      <section className="py-20 bg-white">
        <div className="max-w-6xl mx-auto px-4">
          <div className="text-center mb-16">
            <h2 className="text-5xl font-bold text-gray-800 mb-6">Deployment Platforms</h2>
            <div className="w-24 h-1 bg-gradient-to-r from-blue-500 to-purple-500 mx-auto mb-8"></div>
            <p className="text-xl text-gray-600">We deploy your AI applications on leading cloud platforms</p>
          </div>
          
          <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-6 gap-6">
            {platforms.map((platform, index) => (
              <div 
                key={index}
                className={`bg-gradient-to-br ${platform.color} text-white p-6 rounded-2xl text-center hover:scale-110 transition-all duration-300 cursor-pointer shadow-lg hover:shadow-xl`}
                style={{ animationDelay: `${index * 0.1}s` }}
              >
                <div className="text-3xl mb-2">{platform.icon}</div>
                <div className="font-bold text-sm">{platform.name}</div>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* Testimonials Section */}
      <section className="py-20 bg-gray-50">
        <div className="max-w-6xl mx-auto px-4">
          <div className="text-center mb-16">
            <h2 className="text-5xl font-bold text-gray-800 mb-6">What Clients Say</h2>
            <div className="w-24 h-1 bg-gradient-to-r from-blue-500 to-purple-500 mx-auto"></div>
          </div>
          
          <div className="grid md:grid-cols-2 gap-8">
            {testimonials.map((testimonial, index) => (
              <div key={index} className="bg-white p-8 rounded-3xl shadow-lg">
                <div className="text-4xl text-blue-500 mb-4">"</div>
                <p className="text-gray-700 text-lg mb-6 italic">{testimonial.text}</p>
                <div className="text-blue-600 font-semibold">- {testimonial.author}</div>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* Contact Section */}
      <section id="contact" className="py-20 bg-gradient-to-br from-gray-800 to-blue-900 text-white">
        <div className="max-w-6xl mx-auto px-4 text-center">
          <div className="mb-16">
            <h2 className="text-5xl font-bold mb-6">Ready to Transform Your Business?</h2>
            <div className="w-24 h-1 bg-gradient-to-r from-yellow-400 to-orange-500 mx-auto mb-8"></div>
            <p className="text-xl opacity-90 max-w-3xl mx-auto">
              Let's discuss how AI can revolutionize your business operations and drive unprecedented growth
            </p>
          </div>
          
          <div className="bg-gradient-to-r from-blue-600 to-purple-700 p-8 md:p-12 rounded-3xl shadow-2xl max-w-4xl mx-auto">
            <div className="grid md:grid-cols-2 gap-8 items-center">
              <div className="space-y-6">
                <div className="flex items-center justify-center md:justify-start space-x-4">
                  <span className="text-3xl">📞</span>
                  
                </div>
                <div className="space-y-3 text-lg">
                  <div className="flex items-center justify-center md:justify-start space-x-3">
                    <span>📧</span>
                    <span>Contact us for consultation</span>
                  </div>
                  <div className="flex items-center justify-center md:justify-start space-x-3">
                    <span>💼</span>
                    <span>Custom solutions for every need</span>
                  </div>
                  <div className="flex items-center justify-center md:justify-start space-x-3">
                    <span>⚡</span>
                    <span>Fast delivery & 24/7 support</span>
                  </div>
                </div>
              </div>
              
              <div className="space-y-4">
                <a 
                  href={whatsappUrl}
                  target="_blank"
                  rel="noopener noreferrer"
                  className="block bg-green-500 hover:bg-green-600 text-white px-8 py-4 rounded-full text-lg font-bold transition-all duration-300 hover:scale-105 shadow-lg"
                >
                  💬 WhatsApp Now
                </a>
                <div className="text-sm opacity-75">
                  Click to start a conversation on WhatsApp
                </div>
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* Footer */}
      <footer className="bg-gray-900 text-white py-12">
        <div className="max-w-6xl mx-auto px-4 text-center">
          <div className="flex items-center justify-center space-x-2 mb-6">
            <span className="text-3xl">🤖</span>
            <span className="text-2xl font-bold">AI Solutions</span>
          </div>
          <p className="text-gray-400 mb-4">
            Transforming businesses with cutting-edge AI technology
          </p>
          <div className="text-sm text-gray-500">
            © 2025 AI Application Development Services. All rights reserved.
          </div>
        </div>
      </footer>

      <style jsx>{`
        @keyframes shimmer {
          0% { transform: translateX(-100%); }
          100% { transform: translateX(100%); }
        }
      `}</style>
    </div>
  );
};

export default AIBusinessWebsite;