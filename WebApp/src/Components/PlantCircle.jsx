import React, { useLayoutEffect } from 'react';
import circle from '../Media/plant-circle.png';
import plus from '../Media/plant-plus.png';

const PlantCircle = ({label, x, y, radius}) => {
  const plantStyle = {
    position: 'absolute',
    left: x - radius,
    top: y - radius,
    width: radius * 2,
    height: radius * 2
  };

  const plusStyle = {
    position: 'absolute',
    left: x - 10,
    top: y - 10,
    width: 20,
    height: 20
  }

  const labelStyle = {
    position: 'absolute',
    width: 'max-content',
    fontFamily: 'Roboto Mono',
    textTransform: 'uppercase',
    fontWeight: 'bold',
    letterSpacing: '6px',
    fontSize: '25px',
    left: x + radius * 0.9,
    top: y - radius * 0.9,
    margin: 0
  }

  return (<div style={{transition: 'all 0.2s ease-in-out'}}>
    <img src={circle} style={plantStyle} />
    <img src={plus} style={plusStyle} />
    <p style={labelStyle}>{label}</p>
  </div>);
}

export default PlantCircle;