import React from 'react';
import watercolorImg from '../Media/watercolor.jpg'

const Watercolor = ({type}) => {
  return (<div>
    <img src={watercolorImg}
         className={"watercolor" + (type === "small" ? " watercolor--small" : "")} 
         draggable={false} />
  </div>)
}

export default Watercolor;