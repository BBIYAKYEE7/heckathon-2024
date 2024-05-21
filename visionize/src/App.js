import React, { useEffect, useRef, useState } from 'react';
import * as THREE from 'three';
import { GLTFLoader } from 'three/examples/jsm/loaders/GLTFLoader';
import Logo from './logo.svg';
import './App.css';

function App() {
  const [scrollPosition, setScrollPosition] = useState(0);
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
      setScrollPosition(currentPosition);
    };

    window.addEventListener("scroll", onScroll);
    return () => window.removeEventListener("scroll", onScroll);
  }, [scrollPosition]);

  useEffect(() => {
    let scene, camera, renderer, model;

    const init = () => {
      scene = new THREE.Scene();
      camera = new THREE.PerspectiveCamera(75, window.innerWidth / window.innerHeight, 0.1, 1000);
      renderer = new THREE.WebGLRenderer({ alpha: true });
      renderer.setSize(window.innerWidth, window.innerHeight);
      modelContainer.current.appendChild(renderer.domElement);

      const loader = new GLTFLoader();
      loader.load('goggles.glb', function (gltf) {
        model = gltf.scene;
        scene.add(model);
        animate();
      }, undefined, function (error) {
        console.error(error);
      });

      camera.position.z = 5;
    };

    const animate = () => {
      requestAnimationFrame(animate);
      if (model) {
        if (scrollPosition > 850 && scrollPosition < 1050) {
          model.rotation.y = (scrollPosition - 850) * 0.01;
        }
        if (model.rotation.y > Math.PI / 2) {
          model.rotation.y = Math.PI / 2;
        }
      }
      renderer.render(scene, camera);
    };

    init();

    return () => {
      if (modelContainer.current) {
        modelContainer.current.removeChild(renderer.domElement);
      }
    };
  }, [scrollPosition]);

  return (
    <div className="App" style={{ backgroundColor: bgColor }}>
      <header className="App-header">
        <img 
          src={Logo} 
          className="App-logo" 
          alt="logo" 
          style={{
            position: 'absolute',
            top: '50%',
            left: '50%',
            transform: `translate(-50%, -50%)`,
            opacity: opacity
          }}
        />
        <div style={{ height: '2000px' }}>
          <p style={{ 
            fontWeight: '600',
            fontSize: '3rem',
            position: 'sticky', 
            top: '40%', 
            opacity: scrollPosition > 100 && scrollPosition < 400 ? 1 : 0, 
            transform: scrollPosition > 100 && scrollPosition < 400 ? 'translateY(0)' : 'translateY(100px)', 
            transition: 'opacity 0.5s, transform 0.5s' 
          }}>Visionize</p>
          <p style={{ fontSize: '2rem', fontWeight: '500', position: 'sticky', top: '45%', opacity: scrollPosition > 410 && scrollPosition < 610 ? 1 : 0, transition: 'opacity 0.5s' }}>다른 세상을 만나보세요</p>
          <p style={{ fontSize: '10vim', fontWeight: '500', position: 'sticky', top: '45%', opacity: scrollPosition > 650 && scrollPosition < 820 ? 1 : 0, transition: 'opacity 0.5s' }}>CRAFTER가 야심차게 준비한 새로운 AR의 기준</p>
          <div ref={modelContainer} style={{ position: 'sticky', top: '50%', opacity: scrollPosition > 850 && scrollPosition < 1050 ? 1 : 0, transition: 'opacity 0.5s' }}></div>
        </div>
      </header>
    </div>
  );
}

export default App;