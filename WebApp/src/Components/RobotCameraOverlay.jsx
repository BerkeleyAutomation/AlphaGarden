import React from 'react';

const RobotCameraOverlay = (props) => {
  if (props.shouldDisplay === true) {
    return (
      <div>
        <div className='robot-stats'>
          <div className='robot-stats__item'>
            <p className='robot-stats__item__label'>Berkeley, CA</p>
          </div>
          <div className='robot-stats__item'>
            <p className='robot-stats__item__label'>Coverage:</p>
            <p className='robot-stats__item__label'>58%</p>
          </div>
          <div className='robot-stats__item'>
            <p className='robot-stats__item__label'>Diversity:</p>
            <p className='robot-stats__item__label'>36%</p>
          </div>
        </div>
      </div>
    )
  } else {
    return null;
  }
}

export default RobotCameraOverlay;