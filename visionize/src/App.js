import React, { useEffect, useState } from 'react'; 
import Logo from './logo.svg';
import './App.css';
import Image from './rp.png';

function App() {
  const [scrollPosition, setScrollPosition] = useState(0);
  const [opacity, setOpacity] = useState(1);
  const [bgColor, setBgColor] = useState('white');

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
        <p style={{
          color: '#000',
          position: 'fixed',
          top: 'calc(50% + 100px)', // 로고의 높이의 절반(50%) + 로고의 높이의 절반(50%) + 여백(50px)
          transform: 'translate(-50%, -50%)',
          animation: 'bounce 2s infinite',
          fontSize: '10vim',
          fontWeight: '500',
          width: 'auto',
        }}>
          2024.05.24. Comming Soon.
        </p>
        <div style={{ height: '10000px' }}>
          <p style={{
            fontWeight: '600',
            fontSize: '3rem',
            position: 'sticky',
            top: '45%',
            opacity: scrollPosition > 100 && scrollPosition < 400 ? 1 : 0,
            transform: scrollPosition > 100 && scrollPosition < 400 ? 'translateY(0)' : 'translateY(100px)',
            transition: 'opacity 0.5s, transform 0.5s'
          }}>Visionize</p>
          <p style={{
            fontSize: '10vim',
            fontWeight: '600',
            position: 'sticky',
            top: '50%',
            opacity: scrollPosition > 550 && scrollPosition < 760 ? 1 : 0,
            transform: scrollPosition > 550 && scrollPosition < 760 ? 'scale(1)' : 'scale(0.5)', // scale 변경
            transition: 'opacity 0.5s, transform 0.5s' // transform 추가
          }}>
            다른 세상을 만나보세요
          </p>
          <p style={{ fontSize: '16vim', fontWeight: '500', position: 'sticky', top: '50%', opacity: scrollPosition > 900 && scrollPosition < 1380 ? 1 : 0, transition: 'opacity 0.5s' }}><code>CRAFTER</code>가 제시하는 새로운 기준의 마우스.</p>
          <p style={{ fontSize: '16vim', fontWeight: '500', position: 'sticky', top: '50%', opacity: scrollPosition > 1580 && scrollPosition < 2080 ? 1 : 0, transition: 'opacity 0.5s' }}>세계최초 OCR과 그림판이 존재하는 마우스</p>
          <p style={{ fontSize: '16vim', fontWeight: '500', position: 'sticky', top: '50%', opacity: scrollPosition > 2180 && scrollPosition < 2580 ? 1 : 0, transition: 'opacity 0.5s' }}>2개의 획기적인 칩 탑제.</p>
          <img src={Image} alt="image" style={{width: '30%', height: 'auto', position: 'sticky', top: '40%', opacity: scrollPosition > 2680 && scrollPosition < 3500 ? 1 : 0, transform: scrollPosition > 2680 && scrollPosition < 3500 ? 'translateY(0)' : 'translateY(100px)', transition: 'opacity 0.5s, transform 0.5s'}}/>
          <p style={{ fontSize: '16vim', fontWeight: '500', position: 'sticky', top: '50%', opacity: scrollPosition > 3600 && scrollPosition < 4000 ? 1 : 0, transition: 'opacity 0.5s' }}>WeMos ESP8266 그리고</p>
          <p style={{
            fontWeight: '600',
            fontSize: '16vim',
            position: 'sticky',
            top: '50%',
            opacity: scrollPosition > 4100 && scrollPosition < 4500 ? 1 : 0,
            transform: scrollPosition > 4100 && scrollPosition < 4500 ? 'translateY(0)' : 'translateY(100px)',
            transition: 'opacity 0.5s, transform 0.5s'
          }}>ESP32로 종이에 그려 인식하는 그림판 까지.</p>
        </div>
      </header>
    </div>
  );
}

export default App;