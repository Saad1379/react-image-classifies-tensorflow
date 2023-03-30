import React, { useState, useRef } from "react";
import * as tf from "@tensorflow/tfjs";
import * as mobilenet from "@tensorflow-models/mobilenet";

function App() {
  const [model, setModel] = useState(null);
  const [image, setImage] = useState(null);
  const [predictions, setPredictions] = useState([]);
  const imgRef = useRef(null);

  const loadModel = async () => {
    const mobilenetModel = await mobilenet.load();
    setModel(mobilenetModel);
  };

  const classifyImage = async () => {
    const imgElement = imgRef.current;
    const tfimg = tf.browser.fromPixels(imgElement);
    const resizedImg = tf.image.resizeBilinear(tfimg, [224, 224]);
    const batchedImg = resizedImg.expandDims(0);
    const predictions = await model.classify(batchedImg);
    setPredictions(predictions);
  };

  const handleImageUpload = (event) => {
    const file = event.target.files[0];
    if (file) {
      const reader = new FileReader();
      reader.onload = () => {
        setImage(reader.result);
      };
      reader.readAsDataURL(file);
    }
  };

  const handleImageLoad = () => {
    classifyImage();
  };

  if (!model) {
    loadModel();
    return <div>Loading model...</div>;
  }

  return (
    <div>
      <input type="file" accept="image/*" onChange={handleImageUpload} />
      {image && <img src={image} onLoad={handleImageLoad} ref={imgRef} />}
      {predictions.map((p, i) => (
        <div key={i}>{`${p.className}: ${p.probability.toFixed(2)}`}</div>
      ))}
    </div>
  );
}

export default App;
