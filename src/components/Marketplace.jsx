// src/components/Marketplace.jsx
import React, { useEffect } from "react";
import image from "../assets/image.png";

const templates = [
  {
    id: 1,
    title: "Cake Shop Template",
    price: 19.99,
    image: image,
    file: "/cake-shop-template.zip", // path in public folder
  },
];

const Marketplace = () => {
  useEffect(() => {
    // Load Razorpay script
    const script = document.createElement("script");
    script.src = "https://checkout.razorpay.com/v1/checkout.js";
    script.async = true;
    document.body.appendChild(script);
  }, []);

  const handleBuy = (template) => {
    if (!window.Razorpay) {
      alert("Razorpay SDK failed to load. Please refresh and try again.");
      return;
    }

    const options = {
      key: "YOUR_RAZORPAY_KEY", // Replace this
      amount: Math.round(template.price * 100),
      currency: "INR",
      name: "React Template Marketplace",
      description: `Purchase of ${template.title}`,
      image: "https://yourlogo.com/logo.png", // Optional
      handler: function (response) {
        alert("✅ Payment successful! ID: " + response.razorpay_payment_id);
        // Auto-download
        const link = document.createElement("a");
        link.href = template.file;
        link.download = template.title + ".zip";
        document.body.appendChild(link);
        link.click();
        document.body.removeChild(link);
      },
      prefill: {
        name: "Saurabh Katti",
        email: "saurabhkatti2991@gmail.com",
        contact: "7507088360",
      },
      notes: {
        template_id: template.id,
      },
      theme: {
        color: "#6366f1",
      },
    };

    const rzp = new window.Razorpay(options);
    rzp.open();
  };

  return (
    <div className="p-6 bg-gray-100 min-h-screen">
      <h1 className="text-4xl font-bold mb-10 text-center text-gray-900">
        ReactJS Templates
      </h1>

      <div className="max-w-6xl mx-auto grid grid-cols-1 sm:grid-cols-2 md:grid-cols-3 gap-8">
        {templates.map((template) => (
          <div
            key={template.id}
            className="bg-white shadow-xl rounded-2xl overflow-hidden transform hover:scale-105 transition duration-300"
          >
            <img
              src={template.image}
              alt={template.title}
              className="w-full h-56 object-cover"
            />
            <div className="p-6">
              <h2 className="text-2xl font-semibold text-gray-800">
                {template.title}
              </h2>
              <p className="text-gray-600 mt-2 text-lg">₹{template.price}</p>
              <button
                onClick={() => handleBuy(template)}
                className="mt-4 w-full bg-indigo-600 hover:bg-indigo-700 text-white py-2 rounded-lg font-medium transition"
              >
                Buy Now
              </button>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
};

export default Marketplace;
