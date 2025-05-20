import React, { useState, useRef, useEffect } from 'react';
// Removed axios as we'll use EventSource with FormData directly or via fetch for initial setup if needed.
// We can use fetch for sending FormData and then EventSource for the response.

const API_URL = 'http://localhost:5001/api/generate'; // Your backend API URL

interface ChatMessage {
  role: 'user' | 'model';
  content: string;
}

function App() {
  const [currentPrompt, setCurrentPrompt] = useState<string>('');
  const [imageFile, setImageFile] = useState<File | null>(null);
  const [imageUrl, setImageUrl] = useState<string>('');
  const [preview, setPreview] = useState<string | null>(null);
  // const [output, setOutput] = useState<string>(''); // Replaced by chatHistory and streamingOutput
  const [isLoading, setIsLoading] = useState<boolean>(false);
  const [error, setError] = useState<string>('');

  const [useCamera, setUseCamera] = useState<boolean>(false);
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  // Chat specific state
  const [isChatMode, setIsChatMode] = useState<boolean>(false);
  const [chatHistory, setChatHistory] = useState<ChatMessage[]>([]);
  const chatContainerRef = useRef<HTMLDivElement>(null); // For scrolling to bottom

  const DEFAULT_INITIAL_PROMPT = "Analyze this artwork, focusing on its style, historical context, and key artistic features.";

  const clearImageState = () => {
    setImageFile(null);
    setImageUrl('');
    setPreview(null);
    if (fileInputRef.current) {
      fileInputRef.current.value = '';
    }
    stopCameraStream();
    setUseCamera(false);
  };

  const handleNewImageSelected = () => {
    setIsChatMode(false);
    setChatHistory([]);
    setCurrentPrompt(DEFAULT_INITIAL_PROMPT); // Set default prompt for new image
  };

  const handleFileChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    if (event.target.files && event.target.files[0]) {
      const file = event.target.files[0];
      clearImageState();
      handleNewImageSelected();
      setImageFile(file);
      const reader = new FileReader();
      reader.onloadend = () => setPreview(reader.result as string);
      reader.readAsDataURL(file);
    }
  };

  const handleImageUrlChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    const url = event.target.value;
    clearImageState();
    handleNewImageSelected();
    setImageUrl(url);
    setPreview(url || null);
  };
  
  const startCamera = async () => {
    clearImageState();
    handleNewImageSelected();
    setUseCamera(true);
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ video: true });
      if (videoRef.current) videoRef.current.srcObject = stream;
    } catch (err) {
      console.error("Error accessing camera: ", err);
      setError("Could not access camera. Please check permissions.");
      setUseCamera(false);
    }
  };

  const stopCameraStream = () => {
    if (videoRef.current && videoRef.current.srcObject) {
      const stream = videoRef.current.srcObject as MediaStream;
      stream.getTracks().forEach(track => track.stop());
      videoRef.current.srcObject = null;
    }
  };

  const captureImage = () => {
    if (videoRef.current && canvasRef.current) {
      const video = videoRef.current;
      const canvas = canvasRef.current;
      canvas.width = video.videoWidth;
      canvas.height = video.videoHeight;
      const context = canvas.getContext('2d');
      if (context) {
        context.drawImage(video, 0, 0, canvas.width, canvas.height);
        canvas.toBlob((blob) => {
          if (blob) {
            const capturedFile = new File([blob], "camera_capture.png", { type: "image/png" });
            clearImageState(); 
            handleNewImageSelected();
            setImageFile(capturedFile);
            setPreview(canvas.toDataURL('image/png'));
            stopCameraStream();
            setUseCamera(false);
          }
        }, 'image/png');
      }
    }
  };

  const handleSubmit = async () => {
    const promptToSubmit = currentPrompt.trim();
    if (!promptToSubmit && !isChatMode) {
        setError('Please provide an initial instruction or question about the artwork.');
        return;
    }
    if (isChatMode && !promptToSubmit) {
        setError('Please enter a message.');
        return;
    }
    if (!imageFile && !imageUrl) {
      setError('Please select an image to analyze.');
      return;
    }

    setError('');
    setIsLoading(true);
    
    const userMessage: ChatMessage = { role: 'user', content: promptToSubmit };
    // If not in chat mode, this is the first message. Otherwise, append.
    const currentMessages = isChatMode ? [...chatHistory, userMessage] : [userMessage];
    setChatHistory(currentMessages);
    setCurrentPrompt(''); 

    const formData = new FormData();
    formData.append('messages', JSON.stringify(currentMessages));

    if (imageFile) formData.append('image_file', imageFile);
    else if (imageUrl) formData.append('image_url', imageUrl);

    try {
        const response = await fetch(API_URL, { method: 'POST', body: formData });

        if (response.ok && response.body) {
            if (!isChatMode) setIsChatMode(true);
            setChatHistory(prev => [...prev, { role: 'model', content: '' }]);
        } else if (!response.ok) {
            let errorMsg = `Error: ${response.status} ${response.statusText}`;
            try { const errData = await response.json(); errorMsg = errData.error || errorMsg; } catch (e) { /* ignore */ }
            setChatHistory(prev => prev.filter(msg => msg.role !== 'user' || msg.content !== userMessage.content));
            throw new Error(errorMsg);
        } else {
            setChatHistory(prev => prev.filter(msg => msg.role !== 'user' || msg.content !== userMessage.content));
            throw new Error('Response body is null');
        }
        
        const reader = response.body.getReader();
        const decoder = new TextDecoder();
        let accumulatedModelResponse = "";

        while (true) {
            const { done, value } = await reader.read();
            if (done) {
                setChatHistory(prev => {
                    const updatedHistory = [...prev];
                    if (updatedHistory.length > 0) {
                        const lastMessageIndex = updatedHistory.length - 1;
                        if (updatedHistory[lastMessageIndex].role === 'model' && updatedHistory[lastMessageIndex].content.trim() === '') {
                            updatedHistory[lastMessageIndex].content = "[SmolArtExpert seems to be at a loss for words! No analysis provided.]";
                        }
                    }
                    return updatedHistory;
                });
                setIsLoading(false);
                break;
            }
            
            const chunk = decoder.decode(value, { stream: true });
            const lines = chunk.split('\n\n').filter(line => line.trim() !== '');

            for (const line of lines) {
                if (line.startsWith('data: ')) {
                    const jsonData = line.substring(6);
                    try {
                        const parsed = JSON.parse(jsonData);
                        if (parsed.token) {
                            accumulatedModelResponse += parsed.token;
                            setChatHistory(prev => {
                                const updatedHistory = [...prev];
                                if (updatedHistory.length > 0 && updatedHistory[updatedHistory.length - 1].role === 'model') {
                                    updatedHistory[updatedHistory.length - 1].content = accumulatedModelResponse;
                                }
                                return updatedHistory;
                            });
                        } else if (parsed.error) {
                            setError(`Stream error: ${parsed.error}`);
                            setIsLoading(false);
                            setChatHistory(prev => prev.filter(msg => !(msg.role === 'model' && msg.content === '')));
                            return;
                        } else if (parsed.event === 'eos') {
                           setIsLoading(false);
                           setChatHistory(prev => {
                                const updatedHistory = [...prev];
                                if (updatedHistory.length > 0) {
                                    const lastMessageIndex = updatedHistory.length - 1;
                                    if (updatedHistory[lastMessageIndex].role === 'model' && updatedHistory[lastMessageIndex].content.trim() === '') {
                                        updatedHistory[lastMessageIndex].content = "[SmolArtExpert seems to be at a loss for words! No analysis provided.]";
                                    }
                                }
                                return updatedHistory;
                           });
                        }
                    } catch (e) { console.error("Failed to parse stream data:", jsonData, e); }
                }
            }
        }
    } catch (err: any) {
        console.error("Error during submit/stream: ", err);
        setError(err.message || 'Failed to get response or stream from server.');
        setIsLoading(false);
        setChatHistory(prev => prev.filter(msg => msg.role !== 'user' || msg.content !== userMessage.content));
        setChatHistory(prev => prev.filter(msg => !(msg.role === 'model' && msg.content === '')));
    }
  };

  useEffect(() => {
    if (chatContainerRef.current) {
      chatContainerRef.current.scrollTop = chatContainerRef.current.scrollHeight;
    }
  }, [chatHistory]);

  // Effect to set initial prompt when an image is first selected
  useEffect(() => {
    if (preview && !isChatMode && chatHistory.length === 0) {
      setCurrentPrompt(DEFAULT_INITIAL_PROMPT);
    }
  }, [preview, isChatMode, chatHistory]);

  return (
    <div className="min-h-screen bg-gray-900 text-white flex flex-col items-center p-4 sm:p-6 font-sans">
      <div className="w-full max-w-3xl bg-gray-800 p-6 sm:p-8 rounded-xl shadow-2xl flex flex-col" style={{ height: 'calc(100vh - 4rem)', maxHeight: '900px' }}>
        <h1 className="text-3xl sm:text-4xl font-bold mb-6 text-center text-sky-400">SmolArtExpert</h1>

        {/* Image Selection Area - shown when no preview or not in chat mode initially */} 
        {!preview && !useCamera && (
          <div className="mb-6 p-6 border-2 border-dashed border-gray-600 rounded-lg text-center">
            <h2 className="text-xl font-semibold text-gray-300 mb-3">Welcome, Art Enthusiast!</h2>
            <p className="text-gray-400 mb-4">Begin your art exploration by selecting an image.</p>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mb-3">
                <div>
                    <label htmlFor="imageUrl" className="block text-sm font-medium text-gray-400 mb-1">Image URL</label>
                    <input
                        type="text" id="imageUrl" value={imageUrl} onChange={handleImageUrlChange} disabled={isLoading || useCamera}
                        className="w-full p-2.5 border border-gray-600 rounded-md bg-gray-700 text-white focus:ring-2 focus:ring-sky-500 focus:border-sky-500 transition-colors"
                        placeholder="Paste image URL"
                    />
                </div>
                <div>
                    <label htmlFor="imageFile" className="block text-sm font-medium text-gray-400 mb-1">Upload Image</label>
                    <input
                        type="file" id="imageFile" ref={fileInputRef} accept="image/*" onChange={handleFileChange} disabled={isLoading || useCamera}
                        className="w-full text-sm text-gray-400 file:mr-4 file:py-2 file:px-3 file:rounded-md file:border-0 file:font-semibold file:bg-sky-500 file:text-white hover:file:bg-sky-600 cursor-pointer"
                    />
                </div>
            </div>
            <button
                onClick={startCamera}
                disabled={isLoading}
                className={`w-full md:w-auto mt-2 py-2 px-4 font-semibold rounded-md transition-colors bg-indigo-500 hover:bg-indigo-600 text-white text-sm`}
            >
                Or Use Camera
            </button>
          </div>
        )}

        {useCamera && (
          <div className="mb-4 p-4 border border-gray-700 rounded-lg bg-gray-850 flex flex-col items-center">
            <video ref={videoRef} autoPlay playsInline className="w-full max-w-md border border-gray-600 rounded-md mb-3" style={{maxHeight: '300px'}}></video>
            <button onClick={captureImage} disabled={isLoading} className="py-2 px-5 bg-green-600 hover:bg-green-700 text-white font-semibold rounded-md transition-colors">
              Capture Image
            </button>
            <button onClick={() => { stopCameraStream(); setUseCamera(false);}} className="mt-2 py-1 px-3 bg-red-500 hover:bg-red-600 text-white text-xs rounded-md">
                Close Camera
            </button>
            <canvas ref={canvasRef} style={{ display: 'none' }}></canvas>
          </div>
        )}

        {/* Image Preview - shown once an image is selected/captured, above chat */} 
        {preview && (
          <div className="mb-4 p-3 border border-gray-700 rounded-lg bg-gray-750 flex justify-center items-center" style={{maxHeight: isChatMode ? '250px' : '400px', transition: 'max-height 0.3s ease-in-out'}}>
            <img src={preview} alt="Artwork Preview" className="max-w-full h-auto rounded-md object-contain" style={{maxHeight: isChatMode ? '230px' : '380px'}} />
          </div>
        )}
        
        {/* Chat/Analysis Area - only shown if there's a preview (meaning image is selected) */} 
        {preview && (
        <>
            <div ref={chatContainerRef} className="flex-grow overflow-y-auto mb-4 space-y-4 pr-2 text-base scroll-smooth">
            {/* Initial message if no chat history yet and not loading first analysis */} 
            {!isChatMode && !isLoading && chatHistory.length === 0 && preview && (
                <div className="text-center py-6 text-gray-400">
                    Ready to analyze the artwork. Hit "Analyze Artwork" or type your specific question below.
                </div>
            )}
            {chatHistory.map((msg, index) => (
                <div key={index} className={`flex ${msg.role === 'user' ? 'justify-end' : 'justify-start'}`}>
                <div
                    className={`max-w-[85%] p-3 rounded-xl shadow-md ${
                    msg.role === 'user'
                        ? 'bg-sky-600 text-white'
                        : 'bg-gray-700 text-gray-200'
                    }`}
                >
                    <p className="whitespace-pre-wrap">{msg.content}</p>
                </div>
                </div>
            ))}
            {isLoading && chatHistory[chatHistory.length-1]?.role === 'user' && (
                <div className="flex justify-start">
                    <div className="max-w-[85%] p-3 rounded-xl shadow-md bg-gray-700 text-gray-200 animate-pulse">
                        <p className="whitespace-pre-wrap text-transparent">SmolArtExpert is thinking...</p>
                    </div>
                </div>
            )}
            </div>

            {/* Input Field and Submit Button - always shown if an image is selected */} 
            <div className="mt-auto pt-4 border-t border-gray-700">
            <div className="flex items-end space-x-2">
                <textarea
                id="currentPrompt"
                value={currentPrompt}
                onChange={(e) => setCurrentPrompt(e.target.value)}
                rows={Math.max(1, Math.min(3, currentPrompt.split('\n').length))}
                className="w-full p-3 border border-gray-600 rounded-lg bg-gray-700 text-white focus:ring-2 focus:ring-sky-500 focus:border-sky-500 transition-colors resize-none text-base"
                placeholder={isChatMode ? "Ask a follow-up question..." : DEFAULT_INITIAL_PROMPT}
                disabled={isLoading || !preview}
                onKeyDown={(e) => {
                    if (e.key === 'Enter' && !e.shiftKey && !isLoading && currentPrompt.trim() && preview) {
                        e.preventDefault();
                        handleSubmit();
                    }
                }}
                />
                <button
                onClick={handleSubmit}
                disabled={isLoading || !currentPrompt.trim() || !preview}
                className="h-full py-3 px-5 bg-sky-500 hover:bg-sky-600 text-white font-semibold rounded-lg disabled:opacity-50 disabled:cursor-not-allowed transition-all duration-150 focus:outline-none focus:ring-2 focus:ring-sky-500 focus:ring-offset-2 focus:ring-offset-gray-800 flex items-center justify-center" style={{minHeight: '52px'}}
                >
                {isLoading && chatHistory[chatHistory.length-1]?.role === 'user' ? (
                    <svg className="animate-spin h-5 w-5 text-white" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
                    <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                    <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                    </svg>
                ) : isChatMode ? 'Send' : 'Analyze Artwork'}
                </button>
            </div>
            </div>
        </> 
        )}
        
        {error && (
          <p className="mt-3 text-red-400 text-center bg-red-900 bg-opacity-40 p-2 rounded-md text-sm">{error}</p>
        )}
      </div>
    </div>
  );
}

export default App; 