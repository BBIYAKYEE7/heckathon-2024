import React, { useState, useEffect, useRef } from 'react';
import { Engine, Scene, ArcRotateCamera, Vector3, HemisphericLight, SceneLoader } from '@babylonjs/core';

function App() {
  const [opacity, setOpacity] = useState(1);
  const [bgColor, setBgColor] = useState('white');
  const modelContainer = useRef(null);

  useEffect(() => {
    document.title = "Visionize";
    const link = document.querySelector("link[rel*='icon']") || document.createElement('link');
    link.type = 'image/x-icon';
    link.rel = 'shortcut icon';
    link.href = Logo;
    document.getElementsByTagName('head')[0].appendChild(link);
  }, []);

  useEffect(() => {
    const onScroll = () => {
      let currentPosition = window.pageYOffset;
      if (currentPosition > 50) {
        setOpacity(0);
        setBgColor('black');
      } else {
        setOpacity(1 - currentPosition / 50);
        setBgColor(`rgb(${255 - currentPosition * 5}, ${255 - currentPosition * 5}, ${255 - currentPosition * 5})`);
      }
    };

    window.addEventListener("scroll", onScroll);
    return () => window.removeEventListener("scroll", onScroll);
  }, []);

  useEffect(() => {
    let engine, scene, camera, model;

    const init = () => {
      if (modelContainer.current) {
        engine = new Engine(modelContainer.current, true);
        scene = new Scene(engine);
        camera = new ArcRotateCamera("camera", -Math.PI / 2, Math.PI / 2, 2, Vector3.Zero(), scene);
        camera.attachControl(engine.getRenderingCanvas(), true);
        new HemisphericLight("light", new Vector3(1, 1, 0), scene);

        SceneLoader.ImportMesh("", "./", "goggles.glb", scene, function (newMeshes) {
          model = newMeshes[0];
        });

        engine.runRenderLoop(function () {
          if (scene) {
            scene.render();
          }
        });
      }
    };

    init();

    return () => {
      if (engine) {
        engine.dispose();
      }
    };
  }, []);

  return (
    <div style={{ opacity: opacity, backgroundColor: bgColor }}>
      <canvas ref={modelContainer} />
      {/* Rest of your component */}
    </div>
  );
}

export default App;