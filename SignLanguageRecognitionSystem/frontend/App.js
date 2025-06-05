import React, { useEffect, useRef, useState } from 'react';
import Webcam from 'react-webcam';
import axios from 'axios';

const App = () => {
  const webcamRef = useRef(null);
  const [predictedChar, setPredictedChar] = useState('');
  const [sentence, setSentence] = useState('');

  const captureAndSend = async () => {
    if (webcamRef.current) {
      const imageSrc = webcamRef.current.getScreenshot();

      try {
        const response = await axios.post('http://localhost:5000/predict', {
          image: imageSrc,
        });

        const char = response.data.prediction;
        setPredictedChar(char);
        if (char !== ' ') {
          setSentence(prev => prev + char);
        } else {
          setSentence(prev => prev + ' ');
        }

      } catch (error) {
        console.error('Prediction error:', error);
      }
    }
  };

  useEffect(() => {
    const interval = setInterval(captureAndSend, 1500); // Every 1.5 second
    return () => clearInterval(interval);
  }, []);

  return (
    <div className="min-h-screen bg-gray-100 flex flex-col items-center p-8">
      <h1 className="text-3xl font-bold mb-6 text-center">Sign Language Recognition</h1>
      <Webcam
        ref={webcamRef}
        screenshotFormat="image/jpeg"
        width={400}
        className="rounded-lg shadow-md mb-4"
      />
      <div className="text-lg font-semibold mb-2">
        Predicted Character: <span className="text-blue-600">{predictedChar}</span>
      </div>
      <div className="text-xl font-bold bg-white p-4 rounded shadow-md w-full max-w-xl">
        Sentence: {sentence}
      </div>
    </div>
  );
};

export default App;
