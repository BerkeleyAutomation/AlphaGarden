import React from 'react';

const RobotCameraOverlay = (props) => {
  if (props.shouldDisplay === true) {
    return (
      <div>
        <div class='robot-stats'>
          <div class='robot-stats__item'>
            <p class='robot-stats__item__label'>Berkeley, CA</p>
          </div>
          <div class='robot-stats__item'>
            <p class='robot-stats__item__label'>Coverage:</p>
            <p class='robot-stats__item__label'>58%</p>
          </div>
          <div class='robot-stats__item'>
            <p class='robot-stats__item__label'>Diversity:</p>
            <p class='robot-stats__item__label'>36%</p>
          </div>
        </div>
      </div>
    )
  } else {
    return null;
  }
}

export default RobotCameraOverlay;