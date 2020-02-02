import React from 'react';

const RobotCameraOverlay = (props) => {
  return (<div class='robot-stats'>
    <div class='robot-stats__item'>
      <p class='robot-stats__item__label fade-in'>Berkeley, CA</p>
    </div>
    <div class='robot-stats__item'>
      <p class='robot-stats__item__label fade-in'>Coverage:</p>
      <p class='robot-stats__item__label fade-in'>58%</p>
    </div>
    <div class='robot-stats__item'>
      <p class='robot-stats__item__label fade-in'>Diversity:</p>
      <p class='robot-stats__item__label fade-in'>36%</p>
    </div>
  </div>)
}

export default RobotCameraOverlay;