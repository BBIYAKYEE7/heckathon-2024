import React, { useEffect, useState } from 'react';
import Logo from './logo.svg';
import './App.css';

function App() {
  const [scrollPosition, setScrollPosition] = useState(0);
  const [opacity, setOpacity] = useState(0);
  const [scale, setScale] = useState(3); // Start from a larger scale for a zoom out effect

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
        setOpacity(1);
        setScale(1); // Scale down to 1 when scrolled more than 50px
      } else {
        setOpacity(currentPosition / 50);
        setScale(1 + (50 - currentPosition) / 50 * 2); // Scale down from 3 to 1 as you scroll
      }
      setScrollPosition(currentPosition);
    };

    window.addEventListener("scroll", onScroll);
    return () => window.removeEventListener("scroll", onScroll);
  }, [scrollPosition]);

  return (
    <div className="App" style={{ backgroundColor: 'black' }}>
      <header className="App-header">
        <img src={Logo} className="App-logo" alt="logo" style={{ opacity: opacity, transform: `scale(${scale})`, position: 'fixed', top: `${scrollPosition}px` }}/>
      </header>
      <div style={{ height: '2000px' }}></div>
    </div>
  );
}

export default App;