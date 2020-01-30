import React, { useLayoutEffect } from 'react';
import circle from '../Media/plant-circle.png';
import dotted from '../Media/dotted.svg'
import plus from '../Media/plant-plus.png';

const PlantCircle = ({label, x, y, radius}) => {
  const plantStyle = {
    position: 'absolute',
    left: x - radius,
    top: y - radius,
    width: radius * 2,
    height: radius * 2
  };

  const dottedStyle = {
    position: 'absolute',
    left: x - (radius * 2.4 / 2),
    top: y - (radius * 2.4 / 2),
    width: radius * 2.4,
    height: radius * 2.4
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
    letterSpacing: '6px',
    fontSize: '25px',
    left: x + radius * 0.2,
    top: y + radius * 0.7,
    margin: 0,
    padding: '10px',
    border: '2px solid white'
  }

  return (<div style={{transition: 'all 0.2s ease-in-out'}}>
    <img src={dotted} style={dottedStyle} />
    <img src={circle} style={plantStyle} />
    <img src={plus} style={plusStyle} />
    <p style={labelStyle}>{label}</p>
  </div>);
}

export default PlantCircle;