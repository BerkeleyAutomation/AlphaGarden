import React, { useLayoutEffect } from 'react';
import circle from '../Media/plant-circle.svg';
import openCircle from '../Media/open-circle.svg'
import plus from '../Media/plant-plus.png';

const PlantCircle = ({label, x, y, radius}) => {

  const plantStyle = {
    position: 'absolute',
    left: x - radius,
    top: y - radius,
    width: radius * 2,
    height: radius * 2
  };

  const openCircleStyle = {
    position: 'absolute',
    left: x - (radius * 2.4 / 2),
    top: y - (radius * 2.4 / 2),
    width: radius * 2.4,
    height: radius * 2.4,
    animation: 'spin 4s linear infinite'
  };

  const plusStyle = {
    position: 'absolute',
    left: x - 10,
    top: y - 10,
    width: 20,
    height: 20
  };

  const labelStyle = {
    position: 'absolute',
    width: 'max-content',
    fontFamily: 'Roboto Mono',
    textTransform: 'uppercase',
    letterSpacing: '6px',
    fontSize: '25px',
    fontWeight: '800',
    left: x + radius * 0.7,
    top: y + radius * 0.7,
    margin: 0
  };

  return (<div style={{transition: 'all 0.2s ease-in-out'}}>
    <style>{`
            @keyframes spin {
                 0% { transform: rotate(0deg); }
                 100% { transform: rotate(360deg); }
            }
        `}</style>
    <img src={openCircle} style={openCircleStyle} />
    <img src={circle} style={plantStyle} />
    <img src={plus} style={plusStyle} />
    <p style={labelStyle}>{label}</p>
  </div>);
}

export default PlantCircle;