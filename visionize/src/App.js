import React, { useEffect, useState } from 'react'; 
import Logo from './logo.svg';
import './App.css';
import Image from './1.png';

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
          Scroll Slowly For More Information
        </p>
        <div style={{ height: '10000px' }}>
          <p style={{
            fontWeight: '600',
            fontSize: '3rem',
            position: 'sticky',
            top: '45%',
            opacity: scrollPosition > 100 && scrollPosition < 700 ? 1 : 0,
            transform: scrollPosition > 100 && scrollPosition < 700 ? 'translateY(0)' : 'translateY(100px)',
            transition: 'opacity 0.5s, transform 0.5s'
          }}>Visionize</p>
          <p style={{
            fontSize: '10vim',
            fontWeight: '600',
            position: 'sticky',
            top: '50%',
            opacity: scrollPosition > 1000 && scrollPosition < 1600 ? 1 : 0,
            transform: scrollPosition > 1000 && scrollPosition < 1600 ? 'scale(1)' : 'scale(0.5)', // scale 변경
            transition: 'opacity 0.5s, transform 0.5s' // transform 추가
          }}>
            다른 세상을 만나보세요
          </p>
          <p style={{ fontSize: '16vim', fontWeight: '500', position: 'sticky', top: '50%', opacity: scrollPosition > 1900 && scrollPosition < 2500 ? 1 : 0, transition: 'opacity 0.5s' }}><code>CRAFTER</code>가 제시하는 새로운 기준의 입력방식.</p>
          <p style={{ fontSize: '16vim', fontWeight: '500', position: 'sticky', top: '50%', opacity: scrollPosition > 2800 && scrollPosition < 3400 ? 1 : 0, transition: 'opacity 0.5s' }}>아이트래킹으로 키보드를 입력하고</p>
          <p style={{ fontSize: '16vim', fontWeight: '500', position: 'sticky', top: '50%', opacity: scrollPosition > 3700 && scrollPosition < 4300 ? 1 : 0, transition: 'opacity 0.5s' }}>고갯짓으로 마우스 스크롤이 포함된.</p>
          <img src={Image} alt="1" style={{ width: '100%', height: 'auto', borderRadius: '20px', position: 'sticky', top: '10%', opacity: scrollPosition > 4600 && scrollPosition < 5200 ? 1 : 0, transform: scrollPosition > 4600 && scrollPosition < 5200 ? 'translateY(0)' : 'translateY(100px)', transition: 'opacity 0.5s, transform 0.5s'}}/>
          <p style={{
            fontSize: '10vim',
            fontWeight: '600',
            position: 'sticky',
            top: '50%',
            opacity: scrollPosition > 5500 && scrollPosition < 6400 ? 1 : 0,
            transform: scrollPosition > 5500 && scrollPosition < 6400 ? 'scale(1)' : 'scale(0.5)', // scale 변경
            transition: 'opacity 0.5s, transform 0.5s' // transform 추가
          }}>
            키보드를 사용할 수 없는 상황에서도 눈으로 편리하게.
          </p>
          <p style={{
            fontWeight: '600',
            fontSize: '3rem',
            position: 'sticky',
            top: '45%',
            opacity: scrollPosition > 6700 && scrollPosition < 7500 ? 1 : 0,
            transform: scrollPosition > 6700 && scrollPosition < 7500 ? 'translateY(0)' : 'translateY(100px)',
            transition: 'opacity 0.5s, transform 0.5s'
          }}>This is Visionize.</p>
          <p style={{ fontSize: '16vim', fontWeight: '500', position: 'sticky', top: '50%', opacity: scrollPosition > 7800 && scrollPosition < 8300 ? 1 : 0, transition: 'opacity 0.5s' }}><code>CRAFTER.</code> We make future.</p>
          <p style={{
            fontSize: '10vim',
            fontWeight: '600',
            position: 'sticky',
            top: '50%',
            opacity: scrollPosition > 8600 && scrollPosition < 10000 ? 1 : 0,
            transform: scrollPosition > 8600 && scrollPosition < 10000 ? 'scale(1)' : 'scale(0.5)', // scale 변경
            transition: 'opacity 0.5s, transform 0.5s' // transform 추가
          }}>
            <code>CRAFTER.</code> Minho Choi, Jaeyoon Kim and Sangyeon Lee.
          </p>
        </div>
      </header>
    </div>
  );
}

export default App;